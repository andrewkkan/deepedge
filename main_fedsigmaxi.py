#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import random
import os
import sys
import datetime
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Callable
from typing_extensions import TypedDict

from utils.options import args_parser_fedsigmaxi as args_parser
from models.client_sigmaxi import LocalClient_sigmaxi as LocalClient 
from models.test import test_img
from models.linRegress import DataLinRegress, lin_reg
from utils.util_datasets import get_datasets
from utils.util_model import get_model, gather_flat_params, gather_flat_states, add_states
from utils.util_hyper import calculate_gradient_ref, calculate_lr_local


if __name__ == '__main__':

    print("python main_fedsigmaxi.py " + subprocess.list2cmdline(sys.argv[1:]) + "\n")
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists('data/models/%s'%(args.store_models)):
        os.makedirs('data/models/%s'%(args.store_models))

    if args.screendump_file:
        sdf = open(args.screendump_file, "a")
        sdf.write(' '.join(sys.argv) + '\n')
        sdf.write(str(datetime.datetime.now()) + '\n\n')

    # load dataset and split users
    dataset_train, dataset_test, dict_users, args, = get_datasets(args)
    # build model
    net_glob = get_model(args)
    net_glob = net_glob.to(args.device)
    print(net_glob)
    net_glob.train()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    local_user, nD_by_user_idx = [], []

    if args.local_bs < 1 and args.num_local_steps < 1:
        # Assume full gradient (full dataset) for each local step / epoch
        if args.local_ep < 1:
            args.local_ep = 1
        args.num_local_steps = args.local_ep
    elif args.local_bs < 1 and args.num_local_steps >= 1:
        if args.local_ep < 1:
            args.local_ep = args.num_local_steps
    elif args.local_bs >= 1 and args.num_local_steps < 1:
        if args.local_ep < 1:
            args.local_ep = 1
    elif args.local_bs >= 1 and args.num_local_steps >= 1:
        if args.local_ep < 1:
            pass # Will be calculated within local client initiation

    for idx in range(args.num_users):
        local_user.append(LocalClient(args=args, net=None, dataset=dataset_train, idxs=dict_users[idx], user_idx=idx))
        nD_by_user_idx.append(len(dict_users[idx]))
    last_update = np.ones(args.num_users) * -1

    m = min(max(int(args.frac * args.num_users), 1), args.num_users)

    class StatsGlob(TypedDict):
        gradient_ref:       torch.Tensor
        delt_w:             torch.Tensor
        lr_local:           float
        num_local_steps:    int
    stats_glob: StatsGlob = {
        'gradient_ref': torch.zeros_like(gather_flat_params(net_glob)),
        'delt_w': torch.zeros_like(gather_flat_params(net_glob)),
        'lr_local': float(args.lr_device), # initialize
        'num_local_steps': int(args.num_local_steps),
    }

    trained_users = set()
    for epoch_idx in range(args.epochs):
        deltw_locals, grad_locals, nD_locals, \
            loss_locals_pretrain, loss_locals_train, loss_locals_posttrain, \
            acc_locals, acc_locals_on_local, \
            xi_est_locals, sigma_est_locals, = \
            [], [], [], [], [], [], [], [], [], []

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        trained_users.update(idxs_users)
        for iu_idx, user_idx in enumerate(idxs_users): # updates that are only needed once per round
            last_update[user_idx] = epoch_idx
            local_user[user_idx].sync_update(
                net = copy.deepcopy(net_glob).to(args.device), 
                gradient_ref = stats_glob['gradient_ref'].clone().to(args.device), 
            )
            _, loss = local_user[user_idx].local_test()
            loss_locals_pretrain.append(loss)
            # Modify the below to adjust batch size, based upon change in number of local steps.
            if args.dynamic_batch_size and local_user[user_idx].num_local_steps != new_num_local_steps:
                local_user[user_idx].adjust_batch_size(
                    stats_glob['num_local_steps'],
                )

        train_done = False
        # The following loop is for updates that are needed multiple syncs per round.
        # Since we are not adjusting lr locals, we do not need multiple syncs per round.
        for sync_idx in range(1): 
            for iu_idx, user_idx in enumerate(idxs_users):
                train_out = local_user[user_idx].train_step(
                    lr_local=stats_glob['lr_local'], 
                    num_local_steps=stats_glob['num_local_steps'],
                )
                if train_out['done']:
                    deltw_locals.append(train_out['delt_w'])
                    acc_l, _ = test_img(local_user[user_idx].net, dataset_train, args, stop_at_batch=16, shuffle=True, device=args.device)
                    loss_locals_train.append(train_out['mean_loss_train'])
                    acc_locals.append(acc_l)
                    acc_locals_on_local.append(train_out['mean_accuracy'])
                    nD_locals.append(nD_by_user_idx[user_idx])
                    xi_est_locals.append(train_out['xi_est'])
                    sigma_est_locals.append(train_out['sigma_est'])
                    train_done = True
            if train_done: # possible early finish
                break

        # delt_w is scaled by the number of data samples per client, in theory
        delt_w: torch.Tensor = ((torch.stack(deltw_locals) * torch.tensor(nD_locals).view(-1,1).to(args.device)) / torch.tensor(nD_locals).to(args.device).sum()).sum(dim=0)
        # xi_est = np.std(xi_est_locals)
        sigma_est = np.mean(sigma_est_locals)
        xi_est = np.mean(xi_est_locals)

        assert(args.lr_server == 1.0) # Let's enforce this value for now.
        add_states(
            model = net_glob, 
            flat_states = args.lr_server * delt_w,
        )

        for iu_idx, user_idx in enumerate(idxs_users):
            local_user[user_idx].sync_update(
                net = copy.deepcopy(net_glob).to(args.device),
            )
            _, loss = local_user[user_idx].local_test()
            loss_locals_posttrain.append(loss)            
        if args.use_grad_for_ref or args.use_grad_for_dlr: 
            # The following spends additional bandwidth and cycles to calculate 
            # the clients true gradients at the end of training.
            for iu_idx, user_idx in enumerate(idxs_users):
                grad_locals.append(local_user[user_idx].calc_gradient())
            grad_from_locals: torch.Tensor = torch.stack(grad_locals).mean(axis=0)

        else:
            # grad_from_deltw is not scaled by number of data samples per client, but rather by 
            # number of local steps and learning rate, to derive the local gradients.
            grad_from_deltw = - torch.stack(deltw_locals).mean(axis=0) / float(stats_glob['lr_local']) / float(stats_glob['num_local_steps'])

        if args.use_grad_for_ref:
            grad_ref_est = grad_from_locals
        else:
            grad_ref_est = grad_from_deltw

        if args.use_grad_for_dlr:
            mean_grad_local = grad_from_locals
        else:
            mean_grad_local = grad_from_deltw            

        grad_ref: torch.Tensor = calculate_gradient_ref(
            grad_current = stats_glob['gradient_ref'], 
            grad_w = grad_ref_est, 
            momentum_alpha = args.grad_ref_alpha, 
            epoch_idx = epoch_idx,
        )
        lr_local: float = calculate_lr_local(
            lr_local = stats_glob['lr_local'],
            mean_grad_local = mean_grad_local,
            grad_ref = stats_glob['gradient_ref'],
            hyper_lrlr = args.hyper_lrlr,
        )
        num_local_steps: int = stats_glob['num_local_steps'] # Temp placeholder, will need to add dynamic logic

        stats_glob_update: StatsGlob = {
            'delt_w':           delt_w,
            'gradient_ref':     grad_ref,
            'lr_local':         lr_local,
            'num_local_steps':  num_local_steps,        
        }
        stats_glob.update(stats_glob_update)

        print(f'Round {epoch_idx}, xi_est = {xi_est}, sigma_est = {sigma_est}, lr_local = {lr_local}')

        # print status
        loss_pretrain_avg = sum(loss_locals_pretrain) / len(loss_locals_pretrain)
        loss_train_avg = sum(loss_locals_train) / len(loss_locals_train)
        loss_posttrain_avg = sum(loss_locals_posttrain) / len(loss_locals_posttrain)

        # optimizer_glob.step(flat_deltw_list=deltw_locals, flat_deltos_list=deltos_locals, nD_list=nD_locals)
        print("Epoch idx = ", epoch_idx, ", Net Glob Norm = ", gather_flat_params(net_glob).norm())
        # Calculate accuracy for each round
        acc_glob, loss_glob = test_img(net_glob, dataset_test, args, shuffle=True, device=args.device)
        acc_loc = sum(acc_locals) / len(acc_locals)
        acc_lloc = 100. * sum(acc_locals_on_local) / len(acc_locals_on_local)

        print(
                'Round {:3d}, Devices participated {:2d}, Average training loss {:.8f}, Average pretrain loss {:.8f}, Average posttrain loss {:.8f}, Central accuracy on global test data {:.3f}, Central loss on global test data {:.3f}, Local accuracy on global train data {:.3f}, Local accuracy on local train data {:.3f}'.\
                format(epoch_idx, m, loss_train_avg, loss_pretrain_avg, loss_posttrain_avg, acc_glob, loss_glob, acc_loc, acc_lloc)
        )
        if args.screendump_file:
            sdf.write(
                'Round {:3d}, Devices participated {:2d}, Average training loss {:.8f}, Average pretrain loss {:.8f}, Average posttrain loss {:.8f}, Central accuracy on global test data {:.3f}, Central loss on global test data {:.3f}, Local accuracy on global train data {:.3f}, Local accuracy on local train data {:.3f}\n'.\
                format(epoch_idx, m, loss_train_avg, loss_pretrain_avg, loss_posttrain_avg, acc_glob, loss_glob, acc_loc, acc_lloc)
            )
            sdf.flush()
        loss_train.append(loss_train_avg)

    # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./log/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args, shuffle=True, device=args.device)
    #print("Training accuracy: {:.2f}".format(acc_train))
    print("\nTesting accuracy on test data: {:.2f}, Testing loss: {:.2f}\n".format(acc_test, loss_test))
    if args.screendump_file:
        sdf.write("\nTesting accuracy on test data: {:.2f}, Testing loss: {:.2f}\n".format(acc_test, loss_test))
        sdf.write(str(datetime.datetime.now()) + '\n')
        sdf.close()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import random
import os
import sys
import datetime
from typing import Dict, List, Optional, Tuple, Union, Callable

from utils.options import args_parser_fedhypgrad
from models.client_HypGrad import LocalClient_HypGrad
from models.test import test_img, test_img_ensem
from models.linRegress import DataLinRegress, lin_reg
from utils.util_datasets import get_datasets
from utils.util_model import get_model, gather_flat_params, gather_flat_states, add_states
from utils.util_hyper_grad import update_hyper_grad, calcuate_lr_global, calculate_global_descent


if __name__ == '__main__':

    # parse args
    args = args_parser_fedhypgrad()
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
    if args.sync_interval < 1:
        args.sync_interval = args.num_local_steps
    num_sync_per_round = int(args.num_local_steps / args.sync_interval)
    if args.num_local_steps % args.sync_interval:
        args.num_local_steps = num_sync_per_round * args.sync_interval
    if args.hypergrad_on:
        # Start at same values
        args.lr_device = args.lr_local_init
        args.lr_global_init = args.lr_local_init
    else:
        args.lr_local_init = args.lr_device 
        args.lr_global_init = args.lr_server

    for idx in range(args.num_users):
        local_user.append(LocalClient_HypGrad(args=args, net=None, dataset=dataset_train, idxs=dict_users[idx], user_idx=idx))
        nD_by_user_idx.append(len(dict_users[idx]))
    last_update = np.ones(args.num_users) * -1

    m = min(max(int(args.frac * args.num_users), 1), args.num_users)
    stats_glob = {
        'desc_global': torch.zeros_like(gather_flat_params(net_glob)),
        'delt_w': torch.zeros_like(gather_flat_params(net_glob)),
    }
    hyper_grad_vals: Dict[str, float] = { 
        'lr_global': args.lr_global_init,
        'lr_local': args.lr_local_init,
    }
    hyper_grad_lr: Dict[str, float] = {
        'lr_global': args.hyper_lr,
        'lr_local': args.hyper_lr, 
    }
    for epoch_idx in range(args.epochs):
        deltw_locals, grad_locals, nD_locals, loss_locals, acc_locals, acc_locals_on_local = [], [], [], [], [], []
        hyper_grad_local_agg: List[Dict[str, float]] = []

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for iu_idx, user_idx in enumerate(idxs_users): # updates that are only needed once per round
            last_update[user_idx] = epoch_idx
            local_user[user_idx].sync_update(
                net = copy.deepcopy(net_glob).to(args.device), 
                desc_glob = stats_glob['desc_global'].clone().to(args.device), 
            )
        if args.hypergrad_on:
            hyper_grad_vals['lr_local'] = hyper_grad_vals['lr_global'] # lr_global just means starting lr_local for each round
        else:
            hyper_grad_vals['lr_local'] = args.lr_local_init
        for sync_idx in range(num_sync_per_round): # updates that are needed multiple syncs per round
            for iu_idx, user_idx in enumerate(idxs_users):
                local_user[user_idx].sync_update(
                    lr_local = hyper_grad_vals['lr_local'],
                )
                train_results = local_user[user_idx].train()
                try:
                    hyper_grad_local = train_results['hyper_grad']
                except:
                    assert(sync_idx == num_sync_per_round - 1) # Expected exception, unless the assert condition is broken
                    # print("Unexpected error when unpacking train_results in sync_interval loop!!")
                    train_out, loss, acc_ll = train_results
                    deltw_locals.append(copy.deepcopy(train_out['delt_w']))
                    acc_l, _ = test_img(local_user[user_idx].net, dataset_train, args, stop_at_batch=16, shuffle=True, device=args.device)
                    loss_locals.append(loss)
                    acc_locals.append(acc_l)
                    acc_locals_on_local.append(acc_ll)
                    nD_locals.append(nD_by_user_idx[user_idx])
                    # print("Epoch idx = ", epoch_idx, ", User idx = ", user_idx, ", Loss = ", loss, ", Net norm = ", gather_flat_params(local_user[user_idx].net).norm())
                else:
                    hyper_grad_local_agg.append(hyper_grad_local)
            if args.hypergrad_on:
                update_hyper_grad(hyper_grad_vals, hyper_grad_lr, hyper_grad_local_agg)
            new_lr_local = hyper_grad_vals['lr_local']
            print(f"Round = {epoch_idx}, LR Local = {new_lr_local} \n")

        delt_w = ((torch.stack(deltw_locals) * torch.tensor(nD_locals).view(-1,1).to(args.device)) / torch.tensor(nD_locals).to(args.device).sum()).sum(dim=0)
        delt_w_normalized = delt_w / float(args.num_local_steps)
        assert(args.lr_server == 1.0) # Let's enforce this value for now.
        add_states(
            model = net_glob, 
            flat_states = args.lr_server * delt_w,
        )
        desc_glob: torch.Tensor = calculate_global_descent(
            desc_current = stats_glob['desc_global'], 
            delt_w = delt_w_normalized, 
            momentum_alpha = args.glob_descent_momentum, 
            epoch_idx = epoch_idx,
        )
        lr_global: float = calcuate_lr_global(
            lr_global_current = hyper_grad_vals['lr_global'], 
            hlr = hyper_grad_lr['lr_global'], 
            desc_ref = stats_glob['desc_global'], 
            delt_w = delt_w_normalized,
        )
        stats_glob.update({
            'delt_w':       delt_w,
            'desc_global':  desc_glob,
        })
        hyper_grad_vals.update({
            'lr_global':    lr_global,
        })

        # print status
        loss_avg = sum(loss_locals) / len(loss_locals)
        # optimizer_glob.step(flat_deltw_list=deltw_locals, flat_deltos_list=deltos_locals, nD_list=nD_locals)
        print("Epoch idx = ", epoch_idx, ", Net Glob Norm = ", gather_flat_params(net_glob).norm())
        # Calculate accuracy for each round
        acc_glob, loss_glob = test_img(net_glob, dataset_test, args, shuffle=True, device=args.device)
        acc_loc = sum(acc_locals) / len(acc_locals)
        acc_lloc = 100. * sum(acc_locals_on_local) / len(acc_locals_on_local)

        print(
                'Round {:3d}, Devices participated {:2d}, Average training loss {:.8f}, Central accuracy on global test data {:.3f}, Central loss on global test data {:.3f}, Local accuracy on global train data {:.3f}, Local accuracy on local train data {:.3f}'.\
                format(epoch_idx, m, loss_avg, acc_glob, loss_glob, acc_loc, acc_lloc)
        )
        if args.screendump_file:
            sdf.write(
                'Round {:3d}, Devices participated {:2d}, Average training loss {:.8f}, Central accuracy on global test data {:.3f}, Central loss on global test data {:.3f}, Local accuracy on global train data {:.3f}, Local accuracy on local train data {:.3f}\n'.\
                format(epoch_idx, m, loss_avg, acc_glob, loss_glob, acc_loc, acc_lloc)
            )
            sdf.flush()
        loss_train.append(loss_avg)

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







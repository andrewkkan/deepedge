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

from utils.options import args_parser
from models.client import LocalClientMIME
from models.test import test_img, test_img_ensem
from models.sdlbfgs_fed import SdLBFGS_FedBLA, gather_flat_params, gather_flat_states, add_states, net_params_halper
from models.adaptive_sgd import Adaptive_SGD
from models.linRegress import DataLinRegress, lin_reg
from utils.prepare_datasets import get_datasets
from utils.prepare_model import get_model


from IPython import embed

if __name__ == '__main__':

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
    for idx in range(args.num_users):
        local_user.append(LocalClientMIME(args=args, net=None, dataset=dataset_train, idxs=dict_users[idx], user_idx=idx))
        nD_by_user_idx.append(len(dict_users[idx]))
    last_update = np.ones(args.num_users) * -1

    m = min(max(int(args.frac * args.num_users), 1), args.num_users)
    stats_glob = {
        'delt_w':   torch.zeros_like(gather_flat_params(net_glob)),
        'mom1':     torch.zeros_like(gather_flat_params(net_glob)),
        'mom2':     torch.zeros_like(gather_flat_params(net_glob)),
        'control':  torch.zeros_like(gather_flat_params(net_glob)),
    }
    for epoch_idx in range(args.epochs):
        deltw_locals, grad_locals, nD_locals, loss_locals, acc_locals, acc_locals_on_local = [], [], [], [], [], []

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for iu_idx, user_idx in enumerate(idxs_users):
            local_user[user_idx].stats_update(
                net = copy.deepcopy(net_glob).to(args.device), 
                mom1 = stats_glob['mom1'].clone().to(args.device), 
                mom2 = stats_glob['mom2'].clone().to(args.device),
                control = stats_glob['control'].clone().to(args.device),
            )
            last_update[user_idx] = epoch_idx
            stats_dict, loss, acc_ll = local_user[user_idx].train()
            deltw_locals.append(copy.deepcopy(stats_dict['delt_w']))
            grad_locals.append(copy.deepcopy(stats_dict['grad']))
            acc_l, _ = test_img(local_user[user_idx].net, dataset_train, args, stop_at_batch=16, shuffle=True, device=args.device)
            loss_locals.append(loss)
            acc_locals.append(acc_l)
            acc_locals_on_local.append(acc_ll)
            nD_locals.append(nD_by_user_idx[user_idx])
            # print("Epoch idx = ", epoch_idx, ", User idx = ", user_idx, ", Loss = ", loss, ", Net norm = ", gather_flat_params(local_user[user_idx].net).norm())
            local_user[user_idx].del_stats()

        # optimizer_glob.step(flat_deltw_list=deltw_locals, flat_deltos_list=deltos_locals, nD_list=nD_locals)
        print("Epoch idx = ", epoch_idx, ", Net Glob Norm = ", gather_flat_params(net_glob).norm())

        # Calculate accuracy for each round
        acc_glob, loss_glob = test_img(net_glob, dataset_test, args, shuffle=True, device=args.device)
        acc_loc = sum(acc_locals) / len(acc_locals)
        acc_lloc = 100. * sum(acc_locals_on_local) / len(acc_locals_on_local)

        grad_scaled_avg = ((torch.stack(grad_locals) * torch.tensor(nD_locals).view(-1,1).to(args.device)) / torch.tensor(nD_locals).to(args.device).sum()).sum(dim=0)
        delt_w = ((torch.stack(deltw_locals) * torch.tensor(nD_locals).view(-1,1).to(args.device)) / torch.tensor(nD_locals).to(args.device).sum()).sum(dim=0)
        mom1 = args.adaptive_b1 * stats_glob['mom1'] + (1.0 - args.adaptive_b1) * grad_scaled_avg
        mom2 = args.adaptive_b2 * stats_glob['mom2'] + (1.0 - args.adaptive_b2) * grad_scaled_avg * grad_scaled_avg
        stats_glob = {
            'delt_w':   delt_w,
            'mom1':     mom1,
            'mom2':     mom2,
            'control':  grad_scaled_avg,
        }
        add_states(net_glob, args.lr_server * stats_glob['delt_w'])
        # print status
        loss_avg = sum(loss_locals) / len(loss_locals)
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

        del deltw_locals[:]
        del grad_locals[:]
        del delt_w
        del grad_scaled_avg
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()

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


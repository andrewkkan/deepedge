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

from utils.options import args_parser
from models.client import LocalClientK1BFGS
from models.test import test_img, test_img_ensem
from utils.util_datasets import get_datasets
from utils.util_kronecker import initialize_Hmat, initialize_dLdS, initialize_aaT, initialize_abar, update_grads, update_metrics, update_Hmat
from utils.util_model import get_model_k, gather_flat_params, add_states

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
    net_glob = get_model_k(args)
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
        local_user.append(LocalClientK1BFGS(args=args, net=None, dataset=dataset_train, idxs=dict_users[idx], user_idx=idx))
        nD_by_user_idx.append(len(dict_users[idx]))
    last_update = np.ones(args.num_users) * -1

    m = min(max(int(args.frac * args.num_users), 1), args.num_users)
    stats_glob = {
        'delt_w':       torch.zeros_like(gather_flat_params(net_glob)),
        'H_mat':        initialize_Hmat(net_glob),
        'grad_mom':     torch.zeros_like(gather_flat_params(net_glob)),
        'grad':         torch.zeros_like(gather_flat_params(net_glob)),
        'dLdS':         initialize_dLdS(net_glob),
        'S':            initialize_dLdS(net_glob), # same dimensions as dLdS
    }
    for epoch_idx in range(args.epochs):
        loss_locals, acc_locals, acc_locals_on_local = [], [], []

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        nD_total = 0
        for user_idx in idxs_users:
            nD_total += nD_by_user_idx[user_idx]

        stats_round = { # These are metrics that need to be averaged across users per round
            'dLdS':     initialize_dLdS(net_glob),
            'S':        initialize_dLdS(net_glob), # same dimensions as dLdS
            'aaT':      initialize_aaT(net_glob),
            'abar':     initialize_abar(net_glob),
            'delt_w':   torch.zeros_like(gather_flat_params(net_glob)),
            'grad':     torch.zeros_like(gather_flat_params(net_glob)),
        }

        for iu_idx, user_idx in enumerate(idxs_users):
            local_user[user_idx].stats_update(
                net = copy.deepcopy(net_glob).to(args.device), 
                H_mat = stats_glob['H_mat'],  
                mom = stats_glob['grad_mom'].clone().to(args.device), 
            )
            last_update[user_idx] = epoch_idx
            stats_local, loss, acc_ll = local_user[user_idx].train(round_idx = epoch_idx)
            update_grads(stats_round['delt_w'], stats_local['delt_w'], scale = float(nD_by_user_idx[user_idx])/float(nD_total))
            update_grads(stats_round['grad'], stats_local['grad'], scale = float(nD_by_user_idx[user_idx])/float(nD_total))
            update_metrics(stats_round['dLdS'], stats_local['dLdS'], scale = float(nD_by_user_idx[user_idx])/float(nD_total))
            update_metrics(stats_round['S'], stats_local['S'], scale = float(nD_by_user_idx[user_idx])/float(nD_total))
            update_metrics(stats_round['aaT'], stats_local['aaT'], scale = float(nD_by_user_idx[user_idx])/float(nD_total))
            update_metrics(stats_round['abar'], stats_local['abar'], scale = float(nD_by_user_idx[user_idx])/float(nD_total))
            acc_l, _ = test_img(local_user[user_idx].net, dataset_train, args, stop_at_batch = 16, shuffle = True, device = args.device)
            loss_locals.append(loss)
            acc_locals.append(acc_l)
            acc_locals_on_local.append(acc_ll)
            # print("Epoch idx = ", epoch_idx, ", User idx = ", user_idx, ", Loss = ", loss, ", Net norm = ", gather_flat_params(local_user[user_idx].net).norm())
            local_user[user_idx].del_stats()

        if args.momentum_bc_off == True:
            bias_correction_curr = bias_correction_last = 1.0
        else:
            bias_correction_curr = 1. - args.momentum_beta ** (epoch_idx + 1)
            bias_correction_last = 1. - args.momentum_beta ** epoch_idx
        grad_mom = stats_glob['grad_mom'] * bias_correction_last # undo last bias correction before adding momentum
        grad_mom = args.momentum_beta * grad_mom + (1.0 - args.momentum_beta) * stats_round['grad']
        grad_mom = grad_mom / bias_correction_curr # apply bias correction for current round
        # Update H_mat
        H_mat = update_Hmat(
            stats_glob['H_mat'],
            args,
            epoch_idx,
            dLdS_curr   = stats_round['dLdS'],
            dLdS_last   = stats_glob['dLdS'],
            S_curr      = stats_round['S'],
            S_last      = stats_glob['S'],
            aaT         = stats_round['aaT'],
            abar        = stats_round['abar'],
        )
        stats_glob = {
            'delt_w':       stats_round['delt_w'],
            'H_mat':        H_mat,
            'grad_mom':     grad_mom,
            # 'grad':         stats_round['grad'],
            'dLdS':         stats_round['dLdS'],
            'S':            stats_round['S'],
        }
        add_states(net_glob, args.lr_server * stats_glob['delt_w'])

        print("Epoch idx = ", epoch_idx, ", Net Glob Norm = ", gather_flat_params(net_glob).norm())
        # Calculate accuracy for each round
        if args.dataset == "mnist":
            test_acc_glob, test_loss_glob = test_img(net_glob, dataset_test, args, stop_at_batch = 16, shuffle = True, device = args.device)
            train_acc_glob, train_loss_glob = test_img(net_glob, dataset_train, args, stop_at_batch = 16, shuffle = True, device = args.device)
        else:
            test_acc_glob, test_loss_glob = test_img(net_glob, dataset_test, args, stop_at_batch = -1, shuffle = True, device = args.device)
            train_acc_glob, train_loss_glob = test_img(net_glob, dataset_train, args, stop_at_batch = -1, shuffle = True, device = args.device)
        acc_loc = sum(acc_locals) / len(acc_locals)
        acc_lloc = 100. * sum(acc_locals_on_local) / len(acc_locals_on_local)
        loss_avg = sum(loss_locals) / len(loss_locals)
        
        # print status
        print(
                'Round {:3d}, Devices participated {:2d}, Average training loss {:.8f}, Central accuracy on global test data {:.3f}, Central loss on global test data {:.3f}, Central accuracy on global train data {:.3f}, Central loss on global train data {:.3f}, Local accuracy on global train data {:.3f}, Local accuracy on local train data {:.3f}'.\
                format(epoch_idx, m, loss_avg, test_acc_glob, test_loss_glob, train_acc_glob, train_loss_glob, acc_loc, acc_lloc)
        )
        if args.screendump_file:
            sdf.write(
                'Round {:3d}, Devices participated {:2d}, Average training loss {:.8f}, Central accuracy on global test data {:.3f}, Central loss on global test data {:.3f}, Central accuracy on global train data {:.3f}, Central loss on global train data {:.3f}, Local accuracy on global train data {:.3f}, Local accuracy on local train data {:.3f}\n'.\
                format(epoch_idx, m, loss_avg, test_acc_glob, test_loss_glob, train_acc_glob, train_loss_glob, acc_loc, acc_lloc)
            )
            sdf.flush()
        loss_train.append(loss_avg)

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
    test_acc_glob, test_loss_glob = test_img(net_glob, dataset_test, args, shuffle=True, device=args.device)
    #print("Training accuracy: {:.2f}".format(acc_train))
    print("\nTesting accuracy on test data: {:.2f}, Testing loss: {:.2f}\n".format(test_acc_glob, test_loss_glob))
    if args.screendump_file:
        sdf.write("\nTesting accuracy on test data: {:.2f}, Testing loss: {:.2f}\n".format(test_acc_glob, test_loss_glob))
        sdf.write(str(datetime.datetime.now()) + '\n')
        sdf.close()


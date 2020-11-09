#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random
import os
import sys
import datetime

from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, generic_iid, generic_noniid, cifar100_noniid
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, LeNet5
from models.test import test_img, test_img_ensem
from models.sdlbfgs_fed import SdLBFGS_FedLiSA, gather_flat_params, gather_flat_states, add_states
from models.adaptive_sgd import Adaptive_SGD
from models.linRegress import DataLinRegress, lin_reg

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
        sdf.write(str(sys.argv) + '\n')
        sdf.write(str(datetime.datetime.now()) + '\n\n')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
            # dict_users = mnist_sample_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.noniid_hard)
        args.num_classes = 10
        args.num_channels = 1
        args.task = 'ObjRec'
    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 10
        args.task = 'ObjRec'
    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100)
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 100
        args.task = 'ObjRec'
    elif args.dataset == 'bcct200':
        if args.num_users > 8:
            args.num_users = 8
            print("Warning: limiting number of users to 8.")
        trans_bcct = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=[0.262428402,0.262428402,0.262428402], std=[0.30702864, 0.30702864, 0.30702864])])
        dataset_train = datasets.ImageFolder(root='./data/BCCT200_resized/', transform=trans_bcct)
        dataset_test = dataset_train
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 4
        args.task = 'ObjRec'
    elif args.dataset == 'linregress':
        linregress_numinputs = 50
        args.num_classes = 1
        dataset_train = DataLinRegress(linregress_numinputs, num_outputs=args.num_classes)
        dataset_test = dataset_train
        args.model = 'linregress'
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.task = 'LinReg'
    elif args.dataset == 'linsaddle':
        linregress_numinputs = 2
        args.num_classes = 2
        dataset_train = DataLinRegress(linregress_numinputs, noise_sigma=0.0, num_outputs=args.num_classes)
        dataset_test = dataset_train
        args.model = 'linregress'
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.task = 'LinSaddle'    
    else:
        exit('Error: unrecognized dataset')

    args.img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset != 'mnist':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'lenet5' and args.dataset != 'mnist':
        net_glob = LeNet5(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        net_glob = MLP(dim_in=args.img_size[0]*args.img_size[1]*args.img_size[2], dim_hidden=200,
                       dim_out=args.num_classes,
                       weight_init=args.weight_init, bias_init=args.bias_init).to(args.device)
    elif args.model == 'linregress':    
        net_glob = lin_reg(linregress_numinputs, args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    # The below optimizer can handle FedAvg type of methods, as well as BFGS QN.
    # For FedAdam (TBD), add another optimizer in an if-then clause.
    if args.opt_mode == 0 or args.opt_mode == 1 or args.opt_mode == 2:
        optimizer_glob = SdLBFGS_FedLiSA(
            net=net_glob, 
            lr_server_qn=args.lr_server_qn, 
            lr_server_gd=args.lr_server_gd, 
            lr_device=float(args.lr_device), 
            history_size=args.lbfgs_hist, 
            E_l=float(args.local_ep), 
            nD=float(len(dict_users[0])), 
            Bs=float(args.local_bs),
            opt_mode=args.opt_mode,
            vr_mode=args.vr_mode,
            max_qndn=args.max_qndn,
        )
    elif args.opt_mode == 3:
        optimizer_glob = Adaptive_SGD(
            net=net_glob, 
            lr_server_gd=args.lr_server_gd, 
            lr_device=float(args.lr_device), 
            E_l=float(args.local_ep), 
            nD=float(len(dict_users[0])), 
            Bs=float(args.local_bs),
            adaptive_mode=args.adaptive_mode, 
            tau=args.adaptive_tau,        
            beta1=args.adaptive_b1,        
            beta2=args.adaptive_b2,        
        )
    net_glob = net_glob.to(args.device)
    print(net_glob)
    net_glob.train()

    control_glob = None
    if args.vr_mode > 0:
        control_glob = torch.zeros_like(gather_flat_states(net_glob)).to(args.device)
        if args.vr_mode == 0:
            args.vr_scale = 1.0

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    local_user = []
    for idx in range(args.num_users):
        local_user.append(LocalUpdate(args=args, net=copy.deepcopy(net_glob).to(args.device), dataset=dataset_train, idxs=dict_users[idx], user_idx=idx))
    last_update = np.ones(args.num_users) * -1

    for epoch_idx in range(args.epochs):
        delts_locals, deltc_locals, loss_locals, acc_locals, acc_locals_on_local = [], [], [], [], []

        m = min(max(int(args.frac * args.num_users), 1), args.num_users)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for iu_idx, user_idx in enumerate(idxs_users):

            if args.async_s2d == 1:  # async mode 1 updates after FedAvg (see lines below FedAvg)
                delt_s, delt_c, loss, acc_ll = local_user[user_idx].train()
                delts_locals.append(copy.deepcopy(delt_s))
                if args.vr_mode == 1:
                    deltc_locals.append(copy.deepcopy(delt_c))
            elif args.async_s2d == 2:  # async mode 2 updates before FedAvg
                delt_s, delt_c, loss, acc_ll = local_user[user_idx].train()
                delts_locals.append(copy.deepcopy(delt_s))
                if args.vr_mode == 1:
                    deltc_locals.append(copy.deepcopy(delt_c))
                local_user[user_idx].weight_control_update(net=copy.deepcopy(net_glob).to(args.device), control=control_glob)
            elif args.async_s2d == 0:  # synchronous mode, updates before training
                local_user[user_idx].weight_control_update(net=copy.deepcopy(net_glob).to(args.device), control=control_glob)
                last_update[user_idx] = epoch_idx
                delt_s, delt_c, loss, acc_ll = local_user[user_idx].train()
                delts_locals.append(copy.deepcopy(delt_s))
                if args.vr_mode == 1:
                    deltc_locals.append(copy.deepcopy(delt_c))
            acc_l, _ = test_img(local_user[user_idx].net, dataset_train, args, stop_at_batch=16, shuffle=True, device=args.device)
            loss_locals.append(loss)
            acc_locals.append(acc_l)
            acc_locals_on_local.append(acc_ll)
            # print("Epoch idx = ", epoch_idx, ", User idx = ", user_idx, ", Loss = ", loss, ", Net norm = ", gather_flat_params(local_user[user_idx].net).norm())

            if args.async_s2d == 2:
                last_update[user_idx] = epoch_idx

        deltw_locals, deltos_locals = [], []
        sdk = net_glob.state_dict().keys()
        npk = dict(net_glob.named_parameters()).keys()
        for delts in delts_locals:
            wl, osl = [], []
            offset = 0
            for k in sdk:
                numel = net_glob.state_dict()[k].numel()
                if k in npk:
                    wl.append(delts[offset:offset+numel])
                else:
                    osl.append(delts[offset:offset+numel])
                offset += numel
            deltw_locals.append(torch.cat(wl, 0))
            if osl:
                deltos_locals.append(torch.cat(osl, 0)) 

        if args.vr_mode > 0:
            if args.vr_mode == 1:
                control_glob += torch.stack(deltc_locals).mean(dim=0) * len(delts_locals) / float(args.num_users)
            elif args.vr_mode == 2:
                control_glob += (-torch.stack(deltw_locals).sum(dim=0) / args.lr_device / args.local_ep / (float(len(dict_users[0])) / args.local_bs) -control_glob) / float(args.num_users)
            # if args.screendump_file:
            #     sdf.write("Control norm = " + str(control_glob.norm()) + '\n')
            #     sdf.flush()

        optimizer_glob.step(flat_deltw_list=deltw_locals, flat_deltos_list=deltos_locals)

        print("Epoch idx = ", epoch_idx, ", Net Glob Norm = ", gather_flat_params(net_glob).norm())

        # torch.save(net_glob.state_dict(),"data/models/fedavg_updates/net_glob-async%d-round%d.pt"%(args.async_s2d, epoch_idx))
        if args.async_s2d == 1:  # async mode 1 updates after FedAvg
            for user_idx in idxs_users:
                local_user[user_idx].weight_control_update(net=copy.deepcopy(net_glob).to(args.device), control=control_glob)
                last_update[user_idx] = epoch_idx

        # Calculate accuracy for each round
        acc_glob, _ = test_img(net_glob, dataset_test, args, shuffle=True, device=args.device)
        acc_loc = sum(acc_locals) / len(acc_locals)
        acc_lloc = 100. * sum(acc_locals_on_local) / len(acc_locals_on_local)

        # print status
        loss_avg = sum(loss_locals) / len(loss_locals)
        print(
                'Round {:3d}, Devices participated {:2d}, Average loss {:.8f}, Central accuracy on global test data {:.3f}, Local accuracy on global train data {:.3f}, Local accuracy on local train data {:.3f}'.\
                format(epoch_idx, m, loss_avg, acc_glob, acc_loc, acc_lloc)
        )
        if args.screendump_file:
            sdf.write(
                'Round {:3d}, Devices participated {:2d}, Average loss {:.8f}, Central accuracy on global test data {:.3f}, Local accuracy on global train data {:.3f}, Local accuracy on local train data {:.3f}\n'.\
                format(epoch_idx, m, loss_avg, acc_glob, acc_loc, acc_lloc)
            )
            if args.opt_mode == 0 or args.opt_mode == 1 or args.opt_mode == 2:
                g_norm, d_norm, gdcossim = optimizer_glob.get_debuginfo()
                sdf.write(
                    'G_Norm = {:.5f}, D_Norm = {:.5f}, GDCOSSIM = {:.5f}'.format(g_norm, d_norm, gdcossim)
                )
            sdf.flush()
        loss_train.append(loss_avg)

        print(net_glob.state_dict())
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

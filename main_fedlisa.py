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

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_sample_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, LeNet5
from models.test import test_img, test_img_ensem
from models.sdlbfgs_fed import SdLBFGS_FedLiSA, gather_flat_params


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

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
            # dict_users = mnist_sample_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.noniid_hard)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')

    else:
        exit('Error: unrecognized dataset')
    args.img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'lenet5' and args.dataset == 'cifar':
        net_glob = LeNet5().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        net_glob = MLP(dim_in=args.img_size[0]*args.img_size[1]*args.img_size[2], dim_hidden=200,
                       dim_out=args.num_classes,
                       weight_init=args.weight_init, bias_init=args.bias_init)        
    else:
        exit('Error: unrecognized model')

    optimizer_glob = SdLBFGS_FedLiSA(
        net=net_glob, 
        lr=args.lr_g, 
        lr_l=float(args.lr), 
        history_size=args.lbfgs_hist, 
        E_l=float(args.local_ep), 
        nD=float(len(dict_users[0])), 
        Bs=float(args.local_bs),
        sgd_conjugate=args.sgd_conjugate,
    )
    net_glob = net_glob.to(args.device)
    print(net_glob)
    net_glob.train()

    control_glob = copy.deepcopy(net_glob.state_dict())
    for k,v in control_glob.items():
        control_glob[k] = torch.zeros_like(v)

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    local_user = []
    for idx in range(args.num_users):
        local_user.append(LocalUpdate(args=args, net=copy.deepcopy(net_glob).to(args.device), dataset=dataset_train, idxs=dict_users[idx], LiSA=True))
    last_update = np.ones(args.num_users) * -1

    for epoch_idx in range(args.epochs):
        deltw_locals, deltos_locals, deltc_locals, flatg_locals, loss_locals, acc_locals, acc_locals_on_local = [], [], [], [], [], [], []

        if args.rand_d2s == 0.0: # default
            m = min(max(int(args.frac * args.num_users), 1), args.num_users)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            # idxs_users_start = (epoch_idx*m) % args.num_users
            # idxs_users_end = ((epoch_idx+1)*m) % args.num_users
            # idxs_users = list(range(idxs_users_start,idxs_users_end))
        else:
            if args.rand_d2s == []:
                frac_list = [args.frac]
            else:
                frac_list = args.rand_d2s
            rand_d2s = np.resize(frac_list, args.num_users)
            idxs_users = np.where(np.random.random(args.num_users) < rand_d2s)[0]
            m = len(idxs_users)
            if m == 0:
                m = 1
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        if args.sync2async > 0:
            if epoch_idx > args.sync2async:
                args.async_s2d = 1
            else:
                args.async_s2d = 0

        for iu_idx, user_idx in enumerate(idxs_users):
            if args.async_s2d == 1:  # async mode 1 updates after FedAvg (see lines below FedAvg)
                delt_w, delt_os, delt_c, loss, acc_ll = local_user[user_idx].train_lisa()
                deltw_locals.append(copy.deepcopy(delt_w))
                if delt_os:
                    deltos_locals.append(copy.deepcopy(delt_os))
                deltc_locals.append(copy.deepcopy(delt_c))
            elif args.async_s2d == 2:  # async mode 2 updates before FedAvg
                delt_w, delt_os, delt_c, loss, acc_ll = local_user[user_idx].train_lisa()
                deltw_locals.append(copy.deepcopy(delt_w))
                if delt_os:
                    deltos_locals.append(copy.deepcopy(delt_os))
                deltc_locals.append(copy.deepcopy(delt_c))
                local_user[user_idx].weight_control_update(net=copy.deepcopy(net_glob).to(args.device), control=copy.deepcopy(control_glob))
            elif args.async_s2d == 0:  # synchronous mode, updates before training
                local_user[user_idx].weight_control_update(net=copy.deepcopy(net_glob).to(args.device), control=copy.deepcopy(control_glob))
                last_update[user_idx] = epoch_idx
                # delt_w, delt_os, flat_g, loss, acc_ll = local_user[user_idx].train_grad_only()
                delt_w, delt_os, delt_c, loss, acc_ll = local_user[user_idx].train_lisa()
                deltw_locals.append(copy.deepcopy(delt_w))
                if delt_os is not None:
                    deltos_locals.append(copy.deepcopy(delt_os))
                deltc_locals.append(copy.deepcopy(delt_c))
                # flatg_locals.append(copy.deepcopy(flat_g))
            acc_l, _ = test_img(local_user[user_idx].net, dataset_train, args, stop_at_batch=16, shuffle=True)
            loss_locals.append(loss)
            acc_locals.append(acc_l)
            acc_locals_on_local.append(acc_ll)

            if args.async_s2d == 2:
                last_update[user_idx] = epoch_idx

        optimizer_glob.step(deltw_locals, deltos_locals, flatg_locals, epoch_idx)
        
        if args.scaffold_on == True:
            for k,v in control_glob.items():
                control_glob[k] = torch.zeros_like(v)
            for dc in deltc_locals:
                for k,v in dc.items():
                    control_glob[k] += v / float(args.num_users)

        # torch.save(net_glob.state_dict(),"data/models/fedavg_updates/net_glob-async%d-round%d.pt"%(args.async_s2d, epoch_idx))
        if args.async_s2d == 1:  # async mode 1 updates after FedAvg
            for user_idx in idxs_users:
                local_user[user_idx].weight_control_update(net=copy.deepcopy(net_glob).to(args.device), control=copy.deepcopy(control_glob))
                last_update[user_idx] = epoch_idx

        # Calculate accuracy for each round
        acc_glob, _ = test_img(net_glob, dataset_test, args, shuffle=True)
        acc_loc = sum(acc_locals) / len(acc_locals)
        acc_lloc = 100. * sum(acc_locals_on_local) / len(acc_locals_on_local)

        # print status
        loss_avg = sum(loss_locals) / len(loss_locals)
        print(
                'Round {:3d}, Devices participated {:2d}, Average loss {:.3f}, Central accuracy on global test data {:.3f}, Local accuracy on global train data {:.3f}, Local accuracy on local train data {:.3f}\n\n'.\
                format(epoch_idx, m, loss_avg, acc_glob, acc_loc, acc_lloc)
        )
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
    acc_test, loss_test = test_img(net_glob, dataset_test, args, shuffle=True)
    #print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy on test data: {:.2f}, Testing loss: {:.2f}".format(acc_test, loss_test))

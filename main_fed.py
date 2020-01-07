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

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_sample_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from models.FedMAS import do_MAS_Glob

if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            # dict_users = mnist_iid(dataset_train, args.num_users)
            dict_users = mnist_sample_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
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
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes,
                       weight_init=args.weight_init, bias_init=args.bias_init).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    local_user = []
    for idx in range(args.num_users):
        local_user.append(LocalUpdate(args=args, net=copy.deepcopy(net_glob).to(args.device), dataset=dataset_train, idxs=dict_users[idx]))

    N_omega = 0
    omega_sum = None
    for iter in range(args.epochs):
        w_locals, loss_locals, acc_locals, acc_locals_on_local = [], [], [], []

        if args.rand_d2s == 0.0: # default
            m = min(max(int(args.frac * args.num_users), 1), args.num_users)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
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

        if args.sync_params == True:
            for epoch_idx in range(args.local_ep):
                for batch_idx, (images, labels) in enumerate(local_user[0].ldr_train):
                    w_locals = []
                    for idx in idxs_users:
                        local_user[idx].weight_update(net=copy.deepcopy(net_glob).to(args.device))
                        w, loss, acc_ll = local_user[idx].train_batch(epoch_idx, batch_idx)
                        acc_locals_on_local.append(acc_ll)
                        loss_locals.append(loss)
                        w_locals.append(copy.deepcopy(w))
                    w_glob = FedAvg(w_locals)
                    net_glob.load_state_dict(w_glob)
            for idx in idxs_users:
                local_user[idx].weight_update(net=copy.deepcopy(net_glob).to(args.device))
                acc_l, _ = test_img(local_user[idx].net, dataset_train, args, stop_at_batch=16, shuffle=True)
                acc_locals.append(acc_l)
        else:
            for idx in idxs_users:
                if args.async_s2d == 1: # async mode 1 updates after FedAvg (see lines below FedAvg)
                    w, loss, acc_ll = local_user[idx].train()
                elif args.async_s2d == 2: # async mode 2 updates before FedAvg
                    w, loss, acc_ll = local_user[idx].train()
                    local_user[idx].weight_update(net=copy.deepcopy(net_glob).to(args.device))
                    if args.fedmas > 0.0:
                        omega_sum, N_omega = do_MAS_Glob(args=args, local_user=local_user[idx], net_glob=net_glob,
                                                         omega_sum=omega_sum, N_omega=N_omega)
                elif args.async_s2d == 0:  # synchronous mode, updates before training
                    local_user[idx].weight_update(net=copy.deepcopy(net_glob).to(args.device))
                    if args.fedmas > 0.0:
                        omega_sum, N_omega = do_MAS_Glob(args=args, local_user=local_user[idx], net_glob=net_glob,
                                                         omega_sum=omega_sum, N_omega=N_omega)
                    w, loss, acc_ll = local_user[idx].train()
                acc_l, _ = test_img(local_user[idx].net, dataset_train, args, stop_at_batch=16, shuffle=True)
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(loss)
                acc_locals.append(acc_l)
                acc_locals_on_local.append(acc_ll)

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        if args.async_s2d == 1:  # async mode 1 updates after FedAvg
            for idx in idxs_users:
                local_user[idx].weight_update(net=copy.deepcopy(net_glob).to(args.device))
                if args.fedmas > 0.0:
                    omega_sum, N_omega = do_MAS_Glob(args=args, local_user=local_user[idx], net_glob=net_glob,
                                                     omega_sum=omega_sum, N_omega=N_omega)

        # Calculate accuracy for each round
        acc_glob, _ = test_img(net_glob, dataset_test, args, stop_at_batch=16, shuffle=True)
        acc_loc = sum(acc_locals) / len(acc_locals)
        acc_lloc = 100. * sum(acc_locals_on_local) / len(acc_locals_on_local)

        # print status
        loss_avg = sum(loss_locals) / len(loss_locals)
        print(
                'Round {:3d}, Devices participated {:2d}, Average loss {:.3f}, Central accuracy on global test data {:.3f}, Local accuracy on global train data {:.3f}, Local accuracy on local train data {:.3f}'.\
                format(iter, m, loss_avg, acc_glob, acc_loc, acc_lloc)
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
    acc_test, loss_test = test_img(net_glob, dataset_test, args, stop_at_batch=16, shuffle=True)
    #print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy on test data: {:.2f}, Testing loss: {:.2f}".format(acc_test, loss_test))

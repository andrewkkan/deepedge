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
from models.Nets import MLP, CNNMnist, CNNCifar
from models.test import test_img, test_img_ensem
from models.Fed import FedAvgUpdate
from models.DFAN import DFAN_single
from network.gan import GeneratorA, GeneratorB

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
            # dict_users = mnist_iid(dataset_train, args.num_users)
            dict_users = mnist_sample_iid(dataset_train, args.num_users)
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
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        net_glob = MLP(dim_in=args.img_size[0]*args.img_size[1]*args.img_size[2], dim_hidden=200,
                       dim_out=args.num_classes,
                       weight_init=args.weight_init, bias_init=args.bias_init)
        # optimizer_glob = torch.optim.SGD(net_glob.parameters(), lr=args.lr_S,
        #                                  weight_decay=args.weight_decay, momentum=0.0)
        net_glob = net_glob.to(args.device)
        net_glob.train()
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
    last_update = np.ones(args.num_users) * -1

    leaders_list = []
    for epoch_idx in range(args.epochs):
        net_locals_fedavg, net_ref_fedavg, net_locals_dfan, loss_locals, acc_locals, acc_locals_on_local = [], [], [], [], [], []

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

        if args.sync2async > 0:
            if epoch_idx > args.sync2async:
                args.async_s2d = 1
            else:
                args.async_s2d = 0

        for iu_idx, user_idx in enumerate(idxs_users):
            if args.async_s2d == 1:  # async mode 1 updates after FedAvg (see lines below FedAvg)
                w, loss, acc_ll = local_user[user_idx].train()
                net_local = copy.deepcopy(local_user[user_idx].net)
                net_last_update = copy.deepcopy(local_user[user_idx].last_net)
            elif args.async_s2d == 2:  # async mode 2 updates before FedAvg
                w, loss, acc_ll = local_user[user_idx].train()
                net_local = copy.deepcopy(local_user[user_idx].net)
                net_last_update = copy.deepcopy(local_user[user_idx].last_net)
                local_user[user_idx].weight_update(net=copy.deepcopy(net_glob).to(args.device))
            elif args.async_s2d == 0:  # synchronous mode, updates before training
                local_user[user_idx].weight_update(net=copy.deepcopy(net_glob).to(args.device))
                last_update[user_idx] = epoch_idx
                w, loss, acc_ll = local_user[user_idx].train()
                net_local = copy.deepcopy(local_user[user_idx].net)
                net_last_update = copy.deepcopy(local_user[user_idx].last_net)
            acc_l, _ = test_img(local_user[user_idx].net, dataset_train, args, stop_at_batch=16, shuffle=True)
            loss_locals.append(loss)
            acc_locals.append(acc_l)
            acc_locals_on_local.append(acc_ll)

            # Find new leaders
            fedavg_user = False
            if args.num_leaders > 0:
                if len(leaders_list) == args.num_leaders:
                    oldest_leader_update, oldest_leader_idx = epoch_idx, args.num_leaders + 1
                    for ll_idx, leader in enumerate(leaders_list):
                        if oldest_leader_update > leader['k_minus1']:
                            oldest_leader_update = leader['k_minus1']
                            oldest_leader_idx = ll_idx
                    if oldest_leader_update <= last_update[user_idx]:
                        del leaders_list[oldest_leader_idx]
                        leaders_list.append({
                            'k_minus1':     copy.deepcopy(last_update[user_idx]),
                        })
                        net_locals_fedavg.append(net_local)
                        net_ref_fedavg.append(net_last_update)
                        fedavg_user = True
                elif len(leaders_list) > args.num_leaders:
                    print('Warning: leaders_list length is larger than expected.  Something went wrong!')
                else:
                    leaders_list.append({
                        'k_minus1':     copy.deepcopy(last_update[user_idx]),
                    })
                    net_locals_fedavg.append(net_local)
                    net_ref_fedavg.append(net_last_update)
                    fedavg_user = True

            if fedavg_user == False:
                net_locals_dfan.append(net_local)

            if args.async_s2d == 2:
                last_update[user_idx] = epoch_idx

        w_locals = []
        for nl in net_locals_fedavg:
            w_locals.append(nl.state_dict())
        w_last_update = []
        for nr in net_ref_fedavg:
            w_last_update.append(nr.state_dict())
        w_fedavg = FedAvgUpdate(w_locals, w_last_update, net_glob.state_dict(), float(len(leaders_list)))
        net_glob.load_state_dict(copy.deepcopy(w_fedavg))

        if args.dfan_align_alpha > 0.0 and len(net_locals_dfan) > 0:
            net_glob_copy = copy.deepcopy(net_glob)
            optimizer_glob = torch.optim.SGD(net_glob_copy.parameters(), lr=args.lr_S,
                                             weight_decay=args.weight_decay, momentum=0.0)

            generator = GeneratorA(nz=args.nz, nc=1, img_size=args.img_size)
            optimizer_gen = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
            generator = generator.to(args.device)
            generator.train()
            init_w_gen = copy.deepcopy(generator.state_dict())        

            w_locals, w_last_update = [], []
            for nldf in net_locals_dfan:
                net_glob_copy.load_state_dict(copy.deepcopy(net_glob.state_dict()))
                generator.load_state_dict(copy.deepcopy(init_w_gen))
                DFAN_single(args, nldf, net_glob_copy, generator, (optimizer_glob, optimizer_gen), epoch_idx, reg=True)
                w_locals.append(copy.deepcopy(net_glob_copy.state_dict()))
                w_last_update.append(copy.deepcopy(net_glob.state_dict()))
            w_fedavg = FedAvgUpdate(w_locals, w_last_update, net_glob.state_dict(), float(len(idxs_users)) / args.dfan_align_alpha)
            net_glob.load_state_dict(copy.deepcopy(w_fedavg))

        if args.async_s2d == 1:  # async mode 1 updates after FedAvg
            for user_idx in idxs_users:
                local_user[user_idx].weight_update(net=copy.deepcopy(net_glob).to(args.device))
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

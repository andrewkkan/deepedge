#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import copy
import torch.nn.functional as F
import torch
import random
import os
import sys
import datetime

from utils.options import args_parser
from utils.sampling import mnist_iid, mnist_noniid, generic_iid, generic_noniid, cifar100_noniid
from models.Update import DatasetSplit
from models.Nets import MLP, CNNMnist, CNNCifar, LeNet5
from models.test import test_img, test_img_ensem
from models.sdlbfgs_fed import SdLBFGS_FedLiSA, gather_flat_params, gather_flat_states, add_states

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
    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100)
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 100
    elif args.dataset == 'bcct200':
        trans_bcct = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=[0.26215256, 0.26215256, 0.26215256], std=[0.0468134, 0.0468134, 0.0468134])])
        dataset_train = datasets.ImageFolder(root='./data/BCCT200_resized/', transform=trans_bcct)
        dataset_test = dataset_train
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 4
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
                       weight_init=args.weight_init, bias_init=args.bias_init)        
    else:
        exit('Error: unrecognized model')

    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr_device, momentum=args.momentum)
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

    local_user = []
    for idx in range(args.num_users):
        local_user.append(dict_users[idx])

    for epoch_idx in range(args.epochs):

        # torch.save(net_glob.state_dict(),"data/models/%s/netglob-epoch%s-model%s-dataset%s.pt"%(args.store_models,str(epoch_idx),args.model,args.dataset))
        w_locals, loss_locals, acc_locals, acc_locals_on_local = [], [], [], []

        m = min(max(int(args.frac * args.num_users), 1), args.num_users)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        data_idxs = []
        for idxs in idxs_users:
            data_idxs.extend(local_user[idxs])
        user_data = DataLoader(DatasetSplit(dataset_train, data_idxs), batch_size=args.local_bs, shuffle=True)
        epoch_loss = []
        epoch_accuracy = []
        for local_idx in range(args.local_ep):
            batch_loss = []
            batch_accuracy = []
            for batch_idx, (images, labels) in enumerate(user_data):
                images, labels = images.to(args.device), labels.to(args.device)
                net_glob.zero_grad()
                nn_outputs = net_glob(images)
                nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)
                loss = F.cross_entropy(nn_outputs, labels)
                batch_loss.append(loss.item())
                batch_accuracy.append(sum(nnout_max==labels).float() / len(labels))
                loss.backward()
                optimizer.step()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_accuracy.append(sum(batch_accuracy)/len(batch_accuracy))

        # Calculate accuracy for each round
        acc_glob, _ = test_img(net_glob, dataset_test, args, shuffle=True)

        # print status
        loss_avg = sum(epoch_loss) / len(epoch_loss)
        print(
                'Round {:3d}, Devices participated {:2d}, Average loss {:.3f}, Central accuracy on global test data {:.3f}'.\
                format(epoch_idx, m, loss_avg, acc_glob)
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

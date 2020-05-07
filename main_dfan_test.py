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
from models.DFAN import DFAN_regavg
from models.Fed import FedAvg
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
        trans_mnist = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
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

    generator = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        net_glob = MLP(dim_in=args.img_size[0]*args.img_size[1]*args.img_size[2], dim_hidden=200,
                       dim_out=args.num_classes,
                       weight_init=args.weight_init, bias_init=args.bias_init)
        generator = GeneratorA(nz=args.nz, nc=1, img_size=args.img_size)
        optimizer_glob = torch.optim.SGD(net_glob.parameters(), lr=args.lr_S,
                                         weight_decay=args.weight_decay, momentum=0.9)
        optimizer_gen = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
        net_glob = net_glob.to(args.device)
        generator = generator.to(args.device)
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

    for epoch_idx in range(args.epochs + 1):
        net_locals, loss_locals, acc_locals, acc_locals_on_local = [], [], [], []

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


    net_glob.load_state_dict( torch.load('data/models/%s/netglob-epoch%d-modelmlp-datasetmnist.pt'%(args.store_models, epoch_idx-1)))
    generator.load_state_dict( torch.load('data/models/%s/netgen-epoch%d-datasetmnist.pt'%(args.store_models, epoch_idx-1)))
    for ii in range(10):
        net_locals.append(MLP(dim_in=1024, dim_hidden=200, dim_out=10).to(args.device))
        net_locals[ii].load_state_dict(torch.load('data/models/%s/netlocal%d-epoch%d-modelmlp-datasetmnist.pt'%(args.store_models,ii,epoch_idx)))


    # update global weights
    DFAN_regavg(args, net_locals, net_glob, generator, (optimizer_glob, optimizer_gen), epoch_idx, dataset_test)


    # Calculate accuracy for each round
    acc_glob, _ = test_img(net_glob, dataset_test, args, shuffle=True)
    acc_ensem, _ = test_img_ensem(net_locals, dataset_test, args, shuffle=True)

    # print status
    print(
            'Round {:3d}, Devices participated {:2d}, Average loss {:.3f}, Central accuracy on global test data {:.3f}, Ensemble accuracy on global test data {:.3f}\n\n'.\
            format(epoch_idx, m, loss_avg, acc_glob, acc_ensem)
    )


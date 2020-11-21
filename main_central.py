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
from utils.sampling import mnist_iid, mnist_noniid, generic_iid, generic_noniid, cifar100_noniid, emnist_iid, emnist_noniid
from utils.emnist_dataset import EMNISTDataset_by_write
from models.Update import DatasetSplit
from models.Nets import MLP, CNNMnist, CNNCifar, LeNet5, MNIST_AE
from models.test import test_img, test_img_ensem

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
        args.num_classes = 10
        args.num_channels = 1
        args.task = 'ObjRec'
    elif args.dataset == 'emnist':
        # trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.9635,), (0.1586,))])
        trans_emnist = transforms.Compose([transforms.ToTensor()])
        dataset_train = EMNISTDataset_by_write(train=True, transform=trans_emnist)
        dataset_test = EMNISTDataset_by_write(train=False, transform=trans_emnist)
        args.num_classes = 62
        args.num_channels = 1
        args.task = 'AutoEnc'
        args.model = 'autoenc'
    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        args.num_classes = 10
        args.task = 'ObjRec'
    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100)
        args.num_classes = 100
        args.task = 'ObjRec'
    elif args.dataset == 'bcct200':
        trans_bcct = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=[0.26215256, 0.26215256, 0.26215256], std=[0.0468134, 0.0468134, 0.0468134])])
        dataset_train = datasets.ImageFolder(root='./data/BCCT200_resized/', transform=trans_bcct)
        dataset_test = dataset_train
        args.num_classes = 4
        args.task = 'ObjRec'
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
    elif args.model == 'autoenc':
        net_glob = MNIST_AE(dim_in = args.img_size[0]*args.img_size[1]*args.img_size[2]).to(args.device)
    else:
        exit('Error: unrecognized model')

    net_glob = net_glob.to(args.device)
    optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr_device, betas=(0.9,0.999))
    print(net_glob)
    net_glob.train()

    # training
    loss_train = []
    epoch_loss, epoch_accuracy = [], []

    for epoch_idx in range(args.epochs):
        train_data = DataLoader(dataset_train, batch_size=args.local_bs, shuffle=True)
        batch_loss = []
        batch_accuracy = []
        for batch_idx, (images, labels) in enumerate(train_data):
            images, labels = images.to(args.device), labels.to(args.device)
            net_glob.zero_grad()
            nn_outputs = net_glob(images)
            nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)
            if args.task == 'AutoEnc':
                loss = F.mse_loss(nn_outputs, images, reduction='mean')
                batch_accuracy.append(0.0)
            else:
                loss = F.cross_entropy(nn_outputs, labels)
                batch_accuracy.append(sum(nnout_max==labels).float() / len(labels))
            batch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
        epoch_accuracy.append(sum(batch_accuracy)/len(batch_accuracy))

        # Calculate accuracy for each round
        acc_glob, loss_glob = test_img(net_glob, dataset_test, args, shuffle=True, device=args.device)

        # print status
        loss_avg = epoch_loss[-1]
        print(
                'Round {:3d}, Average loss {:.3f}, Central accuracy on global test data {:.3f}, Central loss on global test data {:.3f}'.\
                format(epoch_idx, loss_avg, acc_glob, loss_glob)
        )
        if args.screendump_file:
            sdf.write(
                'Round {:3d}, Average loss {:.3f}, Central accuracy on global test data {:.3f}, Central loss on global test data {:.3f}\n'.\
                format(epoch_idx, loss_avg, acc_glob, loss_glob)
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
    print("Testing accuracy on test data: {:.2f}, Testing loss: {:.2f}".format(acc_test, loss_test))
    if args.screendump_file:
        sdf.write(
            "Testing accuracy on test data: {:.2f}, Testing loss: {:.2f}\n".format(acc_test, loss_test)
        )
        sdf.flush()
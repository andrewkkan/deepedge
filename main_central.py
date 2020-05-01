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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_sample_iid
from utils.options import args_parser
from models.Update import DatasetSplit
from models.Nets import MLP, CNNMnist, CNNCifar
from models.test import test_img

if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        local_user.append(dict_users[idx])

    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch_idx in range(args.epochs):

        torch.save(net_glob.state_dict(),"data/models/%s/netglob-epoch%s-model%s-dataset%s.pt"%(args.store_models,str(epoch_idx),args.model,args.dataset))

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
                # loss = self.loss_func(nn_outputs, labels)
                loss = F.cross_entropy(nn_outputs, labels)
                if args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        local_idx, batch_idx * len(images), len(user_data.dataset),
                               100. * batch_idx / len(user_data), loss.item()))
                batch_loss.append(loss.item())
                batch_accuracy.append(sum(nnout_max==labels).float() / len(labels))
                loss.backward(retain_graph=True)
                optimizer.step()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_accuracy.append(sum(batch_accuracy)/len(batch_accuracy))

        # Calculate accuracy for each round
        acc_glob, _ = test_img(net_glob, dataset_test, args, stop_at_batch=16, shuffle=True)

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
    acc_test, loss_test = test_img(net_glob, dataset_test, args, stop_at_batch=16, shuffle=True)
    #print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy on test data: {:.2f}, Testing loss: {:.2f}".format(acc_test, loss_test))

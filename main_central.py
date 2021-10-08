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
    dataset_train, dataset_test, dict_users, args, = get_datasets(args)

    # build model
    # build model
    net_glob = get_model(args)
    net_glob = net_glob.to(args.device)
    print(net_glob)
    net_glob.train()

    if args.opt_mode == 3:
        optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr_device, betas=(0.9,0.999))
    elif args.opt_mode == 0:
        optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr_device, momentum=args.momentum)
    elif args.opt_mode == 1:
        optimizer = torch.optim.LBFGS(net_glob.parameters(), lr=args.lr_device, history_size=args.lbfgs_hist)

    # torch.save(net_glob.state_dict(), './data/models/main_central_emnist_AE_' + str(args.seed) + '_init.pt')

    # training
    loss_train = []
    epoch_loss, epoch_accuracy = [], []

    round = 0
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

            #torch.save(net_glob.state_dict(), './data/models/temp/main_central_emnist_round' + str(round) + '.pt')
            round += 1

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

    # torch.save(net_glob.state_dict(), './data/models/main_central_emnist_AE_' + str(args.seed) + '.pt')

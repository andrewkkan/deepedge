#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, net, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.net = net

    def train(self):
        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')

        self.net.train()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        epoch_accuracy = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_accuracy = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()
                nn_outputs = self.net(images)
                nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)                
                loss = self.loss_func(nn_outputs, labels)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                batch_accuracy.append(sum(nnout_max==labels).float() / len(labels))
                loss.backward()
                optimizer.step()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_accuracy.append(sum(batch_accuracy)/len(batch_accuracy))
        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss) , sum(epoch_accuracy)/len(epoch_accuracy)

    def weight_update(self, net):
        self.net = net
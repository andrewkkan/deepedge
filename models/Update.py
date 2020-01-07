#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
import torch.nn.functional as F
from models.FedMAS import do_Omega_Local_Update, calculate_Regularization_Omega

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
        # self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.net = net
        self.net_glob = None
        self._omega_local = []
        self._omega_global = []

    def train(self):
        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')

        self.net.train()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        epoch_accuracy = []
        self.stale_net = copy.deepcopy(self.net.state_dict())
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_accuracy = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()
                nn_outputs = self.net(images)
                nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)
                # loss = self.loss_func(nn_outputs, labels)
                loss = self.CrossEntropyLoss(nn_outputs, labels)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                batch_accuracy.append(sum(nnout_max==labels).float() / len(labels))
                loss.backward(retain_graph=True)
                optimizer.step()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_accuracy.append(sum(batch_accuracy)/len(batch_accuracy))
        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss) , sum(epoch_accuracy)/len(epoch_accuracy)

    def train_batch(self, epoch_idx, batch_idx):
        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')

        self.net.train()
        # train and update
        # optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        if epoch_idx == 0:
            self.train_data_batches = [(images, labels) for batch_idx, (images, labels) in enumerate(self.ldr_train)]

        (images, labels) = self.train_data_batches[batch_idx]
        images, labels = images.to(self.args.device), labels.to(self.args.device)
        self.net.zero_grad()
        nn_outputs = self.net(images)
        nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)
        batch_loss = self.CrossEntropyLoss(nn_outputs, labels)
        batch_loss.backward(retain_graph=True)
        batch_accuracy = sum(nnout_max==labels).float() / len(labels)

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        optimizer.step()

        return self.net.state_dict(), batch_loss.item(), batch_accuracy

    def weight_update(self, net):
        self.net = net
        if self.args.fedmas > 0.0:
            self.net_glob = copy.deepcopy(net)

    def omega_update(self, net, omega_glob, N_omega):
        do_Omega_Local_Update(local_user=self, net=net, omega_glob=omega_glob, N_omega=N_omega)

    @property
    def omega_local(self):
        return self._omega_local

    @omega_local.setter
    def omega_local(self, omega_local):
        self._omega_local = omega_local

    @property
    def omega_global(self):
        return self._omega_global

    @omega_global.setter
    def omega_global(self, omega_global):
        self._omega_global = omega_global

    def CrossEntropyLoss(self, outputs, labels):
        batch_size = outputs.size()[0]            # batch_size
        outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
        outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
        if self._omega_global and self.net_glob:
            regularization_mas = calculate_Regularization_Omega(net_glob=self.net_glob, net_local=self.net, omega_glob=self.omega_global)
        else:
            regularization_mas = 0
        return -torch.sum(outputs)/float(batch_size) +(self.args.fedmas * regularization_mas)

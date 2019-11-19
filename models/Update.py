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
        self._omega = []
        self._omega_global = []

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
                loss.backward(retain_graph=True)
                optimizer.step()
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_accuracy.append(sum(batch_accuracy)/len(batch_accuracy))
        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss) , sum(epoch_accuracy)/len(epoch_accuracy)

    def weight_update(self, net):
        self.net = net

    def omega_update(self, net, omega_glob, N_omega):
        self._omega_global = omega_glob
        if N_omega > 0:
            for idx in range(len(omega_glob)):
                self._omega_global[idx]['weight'] = self._omega_global[idx]['weight'] / float(N_omega)
                self._omega_global[idx]['bias'] = self._omega_global[idx]['bias'] / float(N_omega)

        self._omega = []
        for idx, layer in enumerate(net.modules()):
            if type(layer) == torch.nn.modules.linear.Linear:
                self._omega.append({
                        'idx': idx,
                        'weight': torch.zeros(layer.weight.shape).to(self.args.device).to(self.args.device),
                        'bias': torch.zeros(layer.bias.shape).to(self.args.device).to(self.args.device),
                })
        sample_counter = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            for idx in range(images.shape[0]):
                nn_outputs = net(images[idx])
                nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)
                net.zero_grad()
                nn_outputs.backward(torch.nn.functional.one_hot(nnout_max, nn_outputs.shape[1]).to(torch.float), retain_graph=True)
                net_layer_list = list(net.modules())
                for omega_layer in self._omega:
                    omega_layer['weight'] = omega_layer['weight'] + torch.abs(net_layer_list[omega_layer['idx']].weight.grad.data).to(self.args.device)
                    omega_layer['bias'] = omega_layer['bias'] + torch.abs(net_layer_list[omega_layer['idx']].bias.grad.data).to(self.args.device)
                    sample_counter = sample_counter + 1
        for omega_layer in self._omega:
            omega_layer['weight'] = omega_layer['weight'] / float(sample_counter)
            omega_layer['bias'] = omega_layer['bias'] / float(sample_counter)      

    @property
    def omega(self):
        return self._omega
    

def do_MAS(args, device_idx, local_device, net_glob, omega_sum, N_omega):
    if local_device[device_idx].omega and N_omega > 0:
        for layer_idx in range(len(omega_sum)):
            omega_sum[layer_idx]['weight'] = omega_sum[layer_idx]['weight'] - local_device[device_idx].omega[layer_idx]['weight']
            omega_sum[layer_idx]['bias'] = omega_sum[layer_idx]['bias'] - local_device[device_idx].omega[layer_idx]['bias']
        N_omega = N_omega - 1
    if omega_sum == None:
        local_device[device_idx].omega_update(net=copy.deepcopy(net_glob).to(args.device), omega_glob=None, N_omega=0)
        omega_sum = copy.deepcopy(local_device[device_idx].omega)
    else:
        local_device[device_idx].omega_update(net=copy.deepcopy(net_glob).to(args.device), omega_glob=omega_sum, N_omega=N_omega)
        for layer_idx in range(len(omega_sum)):
            omega_sum[layer_idx]['weight'] = omega_sum[layer_idx]['weight'] + local_device[device_idx].omega[layer_idx]['weight']
            omega_sum[layer_idx]['bias'] = omega_sum[layer_idx]['bias'] + local_device[device_idx].omega[layer_idx]['bias']
    N_omega = N_omega + 1
    return omega_sum, N_omega
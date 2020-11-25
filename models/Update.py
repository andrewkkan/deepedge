#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch.utils.data import DataLoader, Dataset
import copy
import torch.nn.functional as F
from models.sdlbfgs_fed import gather_flat_params, gather_flat_params_with_grad, gather_flat_other_states, gather_flat_grad, gather_flat_states, add_states

from IPython import embed

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
    def __init__(self, args, net, dataset=None, idxs=None, user_idx=0):
        self.args = args
        # self.loss_func = nn.CrossEntropyLoss()
        if self.args.local_bs < 1:
            batch_size = len(idxs)
        else:
            batch_size = self.args.local_bs
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.net = net
        self.last_net = copy.deepcopy(self.net)
        self.net_glob = None
        self._omega_local = []
        self._omega_global = []
        self.control_i = None
        self.control_g = None
        if args.vr_mode != 0:
            self.control_i = torch.zeros_like(gather_flat_states(self.net)).to(args.device)
            self.control_g = torch.zeros_like(gather_flat_states(self.net)).to(args.device)
        self.user_idx = user_idx
        self.embed = False
        self.loss_func = self.CrossEntropyLoss
        if args.task == 'ObjRec':
            self.loss_func = self.CrossEntropyLoss
        elif args.task == 'LinReg':
            self.loss_func = self.MSELoss
        elif args.task == 'LinSaddle':
            self.loss_func = self.MSESaddleLoss
        elif args.task == 'AutoEnc':
            self.loss_func = self.MSELoss

    def train_simple(self):
        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')
        self.last_net = copy.deepcopy(self.net)
        self.net.train()
        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr_device, momentum=self.args.momentum)

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

    def train(self):
        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')
        last_net = copy.deepcopy(self.net)
        self.net.train()

        # train and update
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr_device*self.args.vr_scale, momentum=self.args.momentum)

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
                # loss = self.loss_func(nn_outputs, labels)
                if self.args.task == 'AutoEnc':
                    loss = self.loss_func(nn_outputs, images) / images.shape[-1] / images.shape[-2] / images.shape[-3]
                else:
                    loss = self.loss_func(nn_outputs, labels) 
                if self.args.fedprox > 0.0:
                    loss += self.args.fedprox * (gather_flat_params_with_grad(self.net) - gather_flat_params(last_net)).norm(2)
                if self.args.device_reg_norm2 > 0.0:
                    loss += self.args.device_reg_norm2 * gather_flat_params_with_grad(self.net).norm(2)
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                if not (self.args.task in 'LinReg' or self.args.task in 'LinSaddle' or self.args.task in 'AutoEnc'):
                    batch_accuracy.append(sum(nnout_max==labels).float() / len(labels))
                else:
                    batch_accuracy.append(0)
                loss.backward()
                optimizer.step()
                if self.args.vr_mode != 0:
                    add_states(self.net, -self.args.lr_device * (self.control_g - self.control_i*self.args.vr_scale))

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_accuracy.append(sum(batch_accuracy)/len(batch_accuracy))

        flat_net_states, flat_last_net_states = gather_flat_states(self.net), gather_flat_states(last_net)
        flat_delts = flat_net_states - flat_last_net_states

        flat_deltc = None
        if self.args.vr_mode != 0:
            flat_deltc = ((flat_last_net_states - flat_net_states) / self.args.lr_device / self.args.local_ep / float(batch_idx + 1) - self.control_g) / self.args.vr_scale
            self.control_i += flat_deltc

        del flat_net_states
        del flat_last_net_states
        del optimizer
        del last_net

        return flat_delts, flat_deltc, sum(epoch_loss) / len(epoch_loss) , sum(epoch_accuracy)/len(epoch_accuracy)

    def train_grad_only(self):
        # delt_w, delt_os, loss, acc_ll = self.train_lisa()
        _net = copy.deepcopy(self.net)
        _last_net = copy.deepcopy(self.last_net)
        _net.train()

        _net.zero_grad()
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            nn_outputs = _net(images)
            loss = self.CrossEntropyLoss(nn_outputs, labels)
            loss.backward()
        flat_grad = gather_flat_grad(_net) / float(batch_idx+1)

        flat_other_states = gather_flat_other_states(_net)
        flat_last_other_states = gather_flat_other_states(_last_net)
        if flat_other_states is not None and flat_last_other_states is not None:
            deltos = flat_other_states.sub(flat_last_other_states)
        else:
            deltos = None

        deltw = None
        loss = 0.
        acc_ll = 0.

        return deltw, deltos, flat_grad, loss, acc_ll


    def train_batch(self, epoch_idx, batch_idx):
        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')

        self.net.train()

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

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr_device, momentum=self.args.momentum)
        optimizer.step()

        return self.net.state_dict(), batch_loss.item(), batch_accuracy

    def weight_update(self, net):
        self.net = net

    def weight_control_update(self, net, control):
        self.net = net
        if control is not None:
            self.control_g = copy.deepcopy(control)

    def CrossEntropyLoss(self, outputs, labels):
        batch_size = outputs.size()[0]            # batch_size
        outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
        outputs = outputs[range(batch_size), labels] # labels is 1-hot so the rest softmax outputs go to zero
        return -torch.sum(outputs)/float(batch_size) 

    def MSELoss(self, yhat, y):
        batch_size = yhat.size()[0]
        return torch.sum((yhat - y).pow(2))/float(batch_size)

    def MSESaddleLoss(self, yhat, y):
        batch_size = yhat.size()[0]
        return torch.sum((yhat[:,0] - y[:,0]).pow(2))/float(batch_size) - torch.sum((yhat[:,1] - y[:,1]).pow(2))/float(batch_size)

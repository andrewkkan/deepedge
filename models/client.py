#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch.utils.data import DataLoader, Dataset
import copy
import torch.nn.functional as F
from models.sdlbfgs_fed import gather_flat_params, gather_flat_params_with_grad, gather_flat_other_states, gather_flat_grad, gather_flat_states, add_states, net_params_halper
from models.adaptive_sgd import Adaptive_SGD
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


class LocalClientMIME(object):
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

    def train(self):
        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')
        last_net = copy.deepcopy(self.net)
        self.net.train()
        last_net.train()

        optimizer = Adaptive_SGD(
            net = self.net, 
            lr_server_gd=float(self.args.lr_device), 
            server_opt_mode=self.args.client_opt_mode, 
            tau=self.args.adaptive_tau,        
            beta1=self.args.adaptive_b1,        
            beta2=self.args.adaptive_b2
        )
        optimizer.updateMomVals(self.mom1, self.mom2)

        epoch_loss = []
        epoch_accuracy = []
        grad_ep = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_accuracy = []
            grad_batch = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()
                nn_outputs = self.net(images)
                if self.args.task == 'AutoEnc':
                    loss = self.loss_func(nn_outputs, images) / images.shape[-1] / images.shape[-2] / images.shape[-3]
                else:
                    nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)                
                    loss = self.loss_func(nn_outputs, labels) 
                batch_loss.append(loss.item())
                if not (self.args.task in 'LinReg' or self.args.task in 'LinSaddle' or self.args.task in 'AutoEnc'):
                    batch_accuracy.append(sum(nnout_max==labels).float() / len(labels))
                else:
                    batch_accuracy.append(0)
                loss.backward()
                deltw = gather_flat_grad(self.net)
                last_net.zero_grad()
                _ = last_net(images)
                deltw_ref = gather_flat_grad(last_net)
                if self.args.client_momentum_mode == 0:
                    optimizer.updateMomVals(self.mom1, self.mom2)
                optimizer.step(flat_deltw_list=[deltw-deltw_ref+self.control])
                grad_batch.append(deltw_ref)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_accuracy.append(sum(batch_accuracy)/len(batch_accuracy))
            grad_ep.append(torch.stack(grad_batch).sum(dim=0))
            del grad_batch[:]

        flat_net_states, flat_last_net_states = gather_flat_states(self.net), gather_flat_states(last_net)
        flat_delts = flat_net_states - flat_last_net_states
        flat_grad = torch.stack(grad_ep).sum(dim=0)
        
        del flat_net_states
        del flat_last_net_states
        del last_net
        del grad_ep[:]

        return {'delt_w': flat_delts , 'grad': flat_grad}, sum(epoch_loss) / len(epoch_loss) , sum(epoch_accuracy)/len(epoch_accuracy)

    def stats_update(self, net, mom1, mom2, control):
        self.net = net
        self.mom1 = mom1
        self.mom2 = mom2
        self.control = control

    def del_stats(self):
        del self.net
        del self.mom1
        del self.mom2
        del self.control

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




class LocalClientK1BFGS(object):
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

    def train(self):
        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')
        last_net = copy.deepcopy(self.net)
        self.net.train()
        last_net.train()

        optimizer = Adaptive_SGD(
            net = self.net, 
            lr_server_gd=float(self.args.lr_device), 
            server_opt_mode=self.args.client_opt_mode, 
            tau=self.args.adaptive_tau,        
            beta1=self.args.adaptive_b1,        
            beta2=self.args.adaptive_b2
        )
        optimizer.updateMomVals(self.mom1, self.mom2)

        epoch_loss = []
        epoch_accuracy = []
        grad_ep = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_accuracy = []
            grad_batch = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                self.net.zero_grad()
                nn_outputs = self.net(images)
                if self.args.task == 'AutoEnc':
                    loss = self.loss_func(nn_outputs, images) / images.shape[-1] / images.shape[-2] / images.shape[-3]
                else:
                    nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)                
                    loss = self.loss_func(nn_outputs, labels) 
                batch_loss.append(loss.item())
                if not (self.args.task in 'LinReg' or self.args.task in 'LinSaddle' or self.args.task in 'AutoEnc'):
                    batch_accuracy.append(sum(nnout_max==labels).float() / len(labels))
                else:
                    batch_accuracy.append(0)
                loss.backward()
                deltw = gather_flat_grad(self.net)
                last_net.zero_grad()
                _ = last_net(images)
                deltw_ref = gather_flat_grad(last_net)
                if self.args.client_momentum_mode == 0:
                    optimizer.updateMomVals(self.mom1, self.mom2)
                optimizer.step(flat_deltw_list=[deltw-deltw_ref+self.control])
                grad_batch.append(deltw_ref)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_accuracy.append(sum(batch_accuracy)/len(batch_accuracy))
            grad_ep.append(torch.stack(grad_batch).sum(dim=0))
            del grad_batch[:]

        flat_net_states, flat_last_net_states = gather_flat_states(self.net), gather_flat_states(last_net)
        flat_delts = flat_net_states - flat_last_net_states
        flat_grad = torch.stack(grad_ep).sum(dim=0)
        
        del flat_net_states
        del flat_last_net_states
        del last_net
        del grad_ep[:]

        return {'delt_w': flat_delts , 'grad': flat_grad}, sum(epoch_loss) / len(epoch_loss) , sum(epoch_accuracy)/len(epoch_accuracy)

    def stats_update(self, net, H_mat):
        self.net = net
        self.H_mat = H_mat

    def del_stats(self):
        del self.net 
        del self.H_mat 

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

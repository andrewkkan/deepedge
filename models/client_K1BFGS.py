#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch.utils.data import DataLoader, Dataset
import copy
import torch.nn.functional as F
from utils.util_model import gather_flat_params, gather_flat_params_with_grad, gather_flat_other_states, gather_flat_grad, gather_flat_states, add_states
from utils.util_kronecker import multiply_HgDeltHa, get_s_sgrad, get_aaT_abar, calc_mean_dLdS_S_aaT_abar
from utils.util_lossfn import CrossEntropyLoss, MSELoss, MSESaddleLoss
from utils.util_experimental import simple_linesearch
from utils.util_datasets import DatasetSplit

from IPython import embed


class LocalClient_K1BFGS(object):
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
        self.loss_func = CrossEntropyLoss
        if args.task == 'ObjRec':
            self.loss_func = CrossEntropyLoss
        elif args.task == 'LinReg':
            self.loss_func = MSELoss
        elif args.task == 'LinSaddle':
            self.loss_func = MSESaddleLoss
        elif args.task == 'AutoEnc':
            self.loss_func = MSELoss

    def train(self, round_idx):
        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')
        last_net = copy.deepcopy(self.net)
        self.net.train()
        last_net.train()

        with torch.no_grad():
            clipping_metric = gather_flat_params(last_net).abs().mean()
            # elif self.args.newton_method == "linesearch":
            #     clipping_metric = gather_flat_params(last_net).abs().mean() * self.args.lr_device

        epoch_loss = []
        epoch_accuracy = []
        grad_sum = torch.zeros_like(gather_flat_grad(self.net))
        dLdS_batchlist = []
        aaT_batchlist = []
        abar_batchlist = []
        S_batchlist = []

        if self.args.momentum_bc_off == True:
            bias_correction_curr = bias_correction_last = 1.0
        else:
            bias_correction_curr = 1. - self.args.momentum_beta ** (round_idx + 1)
            bias_correction_last = 1. - self.args.momentum_beta ** round_idx   

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_accuracy = []

            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                self.net.zero_grad()
                nn_outputs, _, _ = self.net(images)
                if self.args.task == 'AutoEnc':
                    loss = self.loss_func(nn_outputs, images) / images.shape[-1] / images.shape[-2] / images.shape[-3]
                    nnout_max = None
                else:
                    nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)                
                    loss = self.loss_func(nn_outputs, labels) 
                batch_loss.append(loss.item())
                if nnout_max is not None:
                    batch_accuracy.append(sum(nnout_max==labels).float() / len(labels))
                else:
                    batch_accuracy.append(0)
                loss.backward()

                with torch.no_grad():
                    deltw = gather_flat_grad(self.net)
                # Add momentum to deltw.  Default is MIME style, i.e. server statistics based
                deltw_mom = self.mom * bias_correction_last 
                deltw_mom = deltw_mom * self.args.momentum_beta + deltw * (1.0 - self.args.momentum_beta)
                deltw_mom /= bias_correction_curr
                # descent = Hg deltw Ha for each layer
                descent = multiply_HgDeltHa(deltw_mom, self.H_mat, self.net.state_dict(), device=self.args.device)
                # descent = deltw_mom
                descent_metric = descent.abs().mean()
                if descent_metric > clipping_metric:
                    descent *= clipping_metric / descent_metric
                if self.args.newton_method == "predetermined":
                    descent *= self.args.lr_device
                elif self.args.newton_method == "linesearch":
                    descent = simple_linesearch(flat_descent=descent, net=self.net, flat_grad=deltw_mom, loss_func=self.loss_func, images_labels=(images, labels), task=self.args.task, curr_loss=loss)
                # Add lr * descent to local net
                add_states(self.net, -descent)

                if iter == (self.args.local_ep - 1):
                    last_net.zero_grad()
                    nn_outputs_ref, s, a = last_net(images)
                    if self.args.task == 'AutoEnc':
                        loss_ref = self.loss_func(nn_outputs_ref, images) / images.shape[-1] / images.shape[-2] / images.shape[-3]
                        nnout_max = None
                    else:
                        nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)                
                        loss_ref = self.loss_func(nn_outputs_ref, labels) 
                    loss_ref.backward()
                    with torch.no_grad():
                        deltw_ref = gather_flat_grad(last_net)
                        s_l, sgrad_l = get_s_sgrad(s)
                        aaT, abar = get_aaT_abar(a)
                    grad_sum += deltw_ref
                    dLdS_batchlist.append(sgrad_l)
                    S_batchlist.append(s_l)
                    aaT_batchlist.append(aaT)
                    abar_batchlist.append(abar)

            with torch.no_grad():
                flat_net_states, flat_last_net_states = gather_flat_states(self.net), gather_flat_states(last_net)
            flat_delts = flat_net_states - flat_last_net_states
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            epoch_accuracy.append(sum(batch_accuracy)/len(batch_accuracy))

        with torch.no_grad():
            flat_net_states, flat_last_net_states = gather_flat_states(self.net), gather_flat_states(last_net)
        flat_delts = flat_net_states - flat_last_net_states
        flat_grad = grad_sum / float(batch_idx + 1)  # This grad has been averaged for the batch size (from loss calculation) as well as the number of batches (here)
        dLdS_mean, S_mean, aaT_mean, abar_mean = calc_mean_dLdS_S_aaT_abar(dLdS_batchlist, S_batchlist, aaT_batchlist, abar_batchlist) # inputs are lists of batches, outputs are lists of per-layer metrics

        self.del_metricList(dLdS_batchlist)
        self.del_metricList(S_batchlist)
        self.del_metricList(aaT_batchlist)
        self.del_metricList(abar_batchlist)

        # flat_delts is simply the change in parameters before and after training.  
        # flat_grad is actually the gradient of the entire dataset from the local client against the model parameters before training (from net_glob)
        upload_stats = {
            'delt_w':   flat_delts , # flat vectorized grad
            'grad':     flat_grad, # flat vectorized grad
            'dLdS':     dLdS_mean, # per-layer list
            'S':        S_mean, # per-layer list
            'aaT':      aaT_mean, # per-layer list
            'abar':     abar_mean, # per-layer list
        }
        return upload_stats, sum(epoch_loss) / len(epoch_loss) , sum(epoch_accuracy)/len(epoch_accuracy)

    def stats_update(self, net, H_mat, mom):
        self.net = net
        self.H_mat = H_mat # Do not modify or delete
        self.mom = mom

    def del_stats(self):
        del self.net 
        # del self.H_mat 
        del self.mom

    def del_metricList(self, l):
        for ll in l:
            for lll in ll:
                del lll
            del ll[:]
        del l[:]
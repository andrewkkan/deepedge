#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
from torch.utils.data import DataLoader, Dataset
import copy
import itertools
import torch.nn.functional as F
from utils.util_model import gather_flat_params, gather_flat_params_with_grad, gather_flat_other_states, gather_flat_grad, gather_flat_states, add_states
from utils.util_lossfn import CrossEntropyLoss, MSELoss, MSESaddleLoss
from utils.util_datasets import DatasetSplit
from models.Nets import NNet

class LocalClient_HypGrad(object):
    def __init__(self, args, net, dataset=None, idxs=None, user_idx=0):
        # To satisfy the assert statement that follows, we want to round up local_ep and 
        # round down local_bs and num_local_steps, when in doubt.
        if args.local_bs < 1 and args.num_local_steps < 1:
            # Assume full gradient (full dataset) for each local step / epoch
            if args.local_ep < 1:
                self.local_ep = 1
            else:
                self.local_ep = args.local_ep
            self.num_local_steps = self.local_ep
            self.local_bs = len(idxs)
        elif args.local_bs < 1 and args.num_local_steps >= 1:
            if args.local_ep < 1:
                self.local_ep = args.num_local_steps
            else:
                self.local_ep = args.local_ep
            self.num_local_steps = args.num_local_steps
            self.local_bs = len(idxs) * self.local_ep // self.num_local_steps
        elif args.local_bs >= 1 and args.num_local_steps < 1:
            if args.local_ep < 1:
                self.local_ep = 1
            else:
                self.local_ep = args.local_ep
            self.local_bs = args.local_bs
            self.num_local_steps = len(idxs) * self.local_ep // self.local_bs
        elif args.local_bs >= 1 and args.num_local_steps >= 1:
            self.local_bs = args.local_bs
            self.num_local_steps = args.num_local_steps
            if args.local_ep < 1:
                self.local_ep = -(-self.num_local_steps * self.local_bs + 0.5) // len(idxs) # Round up
            else:
                self.local_ep = args.local_ep
        if args.lr_local_interval < 1:
            self.lr_local_interval = self.num_local_steps
        else:
            self.lr_local_interval = args.lr_local_interval
        self.hypergrad_on = args.hypergrad_on
        self.early_stop_local = args.early_stop_local
        assert(self.local_bs * self.num_local_steps <= len(idxs) * self.local_ep)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.local_bs, shuffle=True)
        self.net = net
        self.active_state = None
        self.user_idx = user_idx
        self.loss_func = CrossEntropyLoss
        if args.task == 'ObjRec':
            self.loss_func = CrossEntropyLoss
        elif args.task == 'LinReg':
            self.loss_func = MSELoss
        elif args.task == 'LinSaddle':
            self.loss_func = MSESaddleLoss
        elif args.task == 'AutoEnc':
            self.loss_func = MSELoss
        self.args = args

    def train(self) -> Tuple[Dict[str, Union[torch.Tensor, List[float]]], float, float]:
        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')

        with torch.no_grad():
            self.active_state = {
                'net_start_round':      copy.deepcopy(self.net),
                'net_grad':             None, # Placeholder for network to calculate gradient
                'step_loss':            [],
                'step_accuracy':        [],
                'data_iter':            itertools.chain(),
                'lr_local_grad':        [],
            }

        self.net.train()
        for ep_idx in range(self.local_ep):
            self.active_state['data_iter'] = itertools.chain(self.active_state['data_iter'], self.ldr_train)

        for lr_adjust_idx in range(len(self.lr_local)):
            for mini_step_idx in range(self.lr_local_interval):
                try:
                    images, labels = next(self.active_state['data_iter'])
                except StopIteration:
                    print("data_iter is emptied unexpectedly!!")
                    raise
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    self.net.zero_grad()
                    nn_outputs = self.net(images)
                    if self.args.task == 'AutoEnc':
                        loss = self.loss_func(nn_outputs, images) / images.shape[-1] / images.shape[-2] / images.shape[-3]
                        nnout_max = None
                    else:
                        nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)                
                        loss = self.loss_func(nn_outputs, labels) 
                    loss.backward()

                    with torch.no_grad():
                        add_states(self.net, - self.lr_local[lr_adjust_idx] * gather_flat_grad(self.net))

                        self.active_state['step_loss'].append(loss.item())
                        if nnout_max is not None:
                            self.active_state['step_accuracy'].append(sum(nnout_max==labels).float() / len(labels))
                        else:
                            self.active_state['step_accuracy'].append(0)
            flat_grad = self.calc_gradient()
            lr_local_grad: float = ((flat_grad * self.gradient_ref).sum()).item() / float(self.lr_local_interval) # dot prouduct normalized 
            self.active_state['lr_local_grad'].append(lr_local_grad)

        with torch.no_grad():
            flat_net_states, flat_ref_states = gather_flat_states(self.net), gather_flat_states(self.active_state['net_start_round'])
        flat_delts = flat_net_states - flat_ref_states
        mean_loss = sum(self.active_state['step_loss']) / len(self.active_state['step_loss'])
        mean_accuracy = sum(self.active_state['step_accuracy']) / len(self.active_state['step_accuracy'])

        # Run the final net with the entire dataset once, without modifying the net parameters.  
        # Collect gradients at the end.
        flat_grad = self.calc_gradient()

        return ({
            'delt_w': flat_delts, 
            'lr_local_grad': self.active_state['lr_local_grad'], 
            'grad': flat_grad,
            }, 
            mean_loss, 
            mean_accuracy,
        )
        

    def sync_update(self, 
        net: Optional[NNet]=None, 
        gradient_ref: Optional[torch.Tensor]=None, 
        lr_local: Optional[List[float]]=None,
    ) -> None:
        # Do not modify self.value if value is None
        if net is not None:
            self.net = net
        if gradient_ref is not None:
            self.gradient_ref = gradient_ref
        if lr_local is not None:
            self.lr_local = lr_local
            for idx, val in enumerate(self.lr_local):
                if val <= 0.0:
                    self.lr_local[idx] = 0.0

    def calc_gradient(self) -> torch.Tensor:
        self.net.zero_grad() 
        train_loader = self.ldr_train
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            nn_outputs = self.net(images)
            if self.args.task == 'AutoEnc':
                loss = self.loss_func(nn_outputs, images) / images.shape[-1] / images.shape[-2] / images.shape[-3]
                nnout_max = None
            else:
                nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)                
                loss = self.loss_func(nn_outputs, labels) 
            loss.backward()
        flat_grad = gather_flat_grad(self.net) / float(len(train_loader))
        return flat_grad




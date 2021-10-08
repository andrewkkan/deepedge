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
        if args.sync_interval < 1:
            self.sync_interval = self.num_local_steps
        else:
            self.sync_interval = args.sync_interval
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

    def train(self
        ) -> Union[
            Tuple[Dict[torch.Tensor, float], float, float],
            Dict[torch.Tensor, float],
        ]:
        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')

        if not self.active_state:
            self.active_state = {
                'net_start_round':      copy.deepcopy(self.net),
                'net_start_interval':   copy.deepcopy(self.net),
                'step_loss':            [],
                'step_accuracy':        [],
                'data_iter':            itertools.chain(),
                'step_idx':             0,
            }
            self.net.train()
            for ep_idx in range(self.local_ep):
                self.active_state['data_iter'] = itertools.chain(self.active_state['data_iter'], self.ldr_train)
        else:
            self.active_state['net_start_interval'] = copy.deepcopy(self.net)

        for mini_step_idx in range(self.sync_interval):
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
                    add_states(self.net, - self.lr_adapt * gather_flat_grad(self.net))

                    self.active_state['step_loss'].append(loss.item())
                    if nnout_max is not None:
                        self.active_state['step_accuracy'].append(sum(nnout_max==labels).float() / len(labels))
                    else:
                        self.active_state['step_accuracy'].append(0)

                self.active_state['step_idx'] += 1
                if self.active_state['step_idx'] == self.num_local_steps:
                    break

        if self.active_state['step_idx'] == self.num_local_steps:
            # Reached end of round
            with torch.no_grad():
                flat_net_states, flat_ref_states = gather_flat_states(self.net), gather_flat_states(self.active_state['net_start_round'])
            flat_delts = flat_net_states - flat_ref_states
            mean_loss = sum(self.active_state['step_loss']) / len(self.active_state['step_loss'])
            mean_accuracy = sum(self.active_state['step_accuracy']) / len(self.active_state['step_accuracy'])
            self.active_state = None
            return {'delt_w': flat_delts , }, mean_loss, mean_accuracy
        else:
            # Interval sync with server before eaching end of round
            with torch.no_grad():
                flat_net_states, flat_ref_states = gather_flat_states(self.net), gather_flat_states(self.active_state['net_start_interval'])
            flat_delts = flat_net_states - flat_ref_states
            lr_adapt: float = (flat_delts * self.desc_glob / float(self.num_local_steps)).sum() # Dot product
            hyper_grad = {
                'lr_adapt':  lr_adapt,
            }
            return {'hyper_grad':   hyper_grad, }

    def sync_update(self, net: Optional[NNet]=None, desc_glob: Optional[torch.Tensor]=None, lr_adapt: Optional[float]=None):
        # Do not modify self.value if value is None
        if net is not None:
            self.net = net
        if desc_glob is not None:
            self.desc_glob = desc_glob
        if lr_adapt is not None:
            self.lr_adapt = lr_adapt






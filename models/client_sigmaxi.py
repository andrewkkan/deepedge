#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import copy
import itertools
import torch.nn.functional as F
from utils.util_model import gather_flat_params, gather_flat_params_with_grad, gather_flat_other_states, gather_flat_grad, gather_flat_states, add_states
from utils.util_lossfn import CrossEntropyLoss, MSELoss, MSESaddleLoss
from utils.util_datasets import DatasetSplit
from models.Nets import NNet

class LocalClient_sigmaxi(object):
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
        self.dynamic_batch_size = args.dynamic_batch_size
        self.local_bs_adjusted = True # Initialized
        self.est_samples = args.sigma_est_samples
        assert(self.local_bs * self.num_local_steps <= len(idxs) * self.local_ep)
        self.dataset = dataset
        self.data_idxs = idxs
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


    def adjust_batch_size(self, num_local_steps: int) -> None:
        # self.local_ep = number of local epochs per round; this may change.
        # self.num_local_steps = number of steps per round; this may change
        # self.local_bs = batch size i.e. number of samples per step; this may change
        self.num_local_steps = num_local_steps
        if self.num_local_steps < self.local_ep:
            self.local_ep = self.num_local_steps
        total_data_samples = len(self.data_idxs) * self.local_ep
        self.local_bs = total_data_samples // self.num_local_steps
        self.local_bs_adjusted = True


    def train_step(self, lr_local: float,
    ) -> Union[
        Tuple[Dict[str, Union[torch.Tensor, List[float], float]], float, float],
        None,
    ]:
        # Training for each lr_local interval 
        local_steps_per_interval = self.num_local_steps

        if not self.net:
            exit('Error: Device LocalUpdate self.net was not initialized')

        if not self.active_state:
            with torch.no_grad():
                self.active_state = {
                    'net_start_round':      copy.deepcopy(self.net),
                    'step_loss':            [],
                    'step_accuracy':        [],
                    'data_iter':            itertools.chain(),
                    'step_idx':             0,
                    'local_step_gradients': [],
                    'gradient_ref':         [],
                    'sigma_est':            [],
                    'xi_est':               [],
                }
            self.net.train()
            if self.local_bs_adjusted:
                self.ldr_train = DataLoader(
                    DatasetSplit(
                        self.dataset, 
                        self.data_idxs,
                    ), 
                    batch_size=self.local_bs, 
                    shuffle=True
                )
                self.local_bs_adjusted = False
            for ep_idx in range(self.local_ep):
                self.active_state['data_iter'] = itertools.chain(self.active_state['data_iter'], self.ldr_train)

        if lr_local <= 0.0:
            self.active_state['step_idx'] = self.num_local_steps
        else:
            for step_idx in range(local_steps_per_interval):
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
                        local_step_gradient: torch.Tensor = gather_flat_grad(self.net)
                    sigma_est, xi_est = self.estimate_sigma_xi(
                        local_step_gradient, 
                        self.gradient_ref,
                        (images, labels),
                    ) # this needs to happen before update
                    with torch.no_grad():
                        add_states(self.net, - lr_local * local_step_gradient)
                        self.active_state['local_step_gradients'].append(local_step_gradient)
                        self.active_state['sigma_est'].append(sigma_est)
                        self.active_state['xi_est'].append(xi_est)
                        self.active_state['step_loss'].append(loss.item())
                        self.active_state['gradient_ref'].append(self.gradient_ref)
                        if nnout_max is not None:
                            self.active_state['step_accuracy'].append(sum(nnout_max==labels).float() / len(labels))
                        else:
                            self.active_state['step_accuracy'].append(0)

                        if self.args.use_local_gradref_mom:
                            self.gradient_ref = self.gradient_ref * np.power(self.args.grad_ref_alpha, 1.0 / self.num_local_steps)

                    self.active_state['step_idx'] += 1
                    if self.active_state['step_idx'] == self.num_local_steps:
                        break

        if self.active_state['step_idx'] == self.num_local_steps:
            with torch.no_grad():
                flat_net_states, flat_ref_states = gather_flat_states(self.net), gather_flat_states(self.active_state['net_start_round'])
            flat_delts = flat_net_states - flat_ref_states
            mean_loss = sum(self.active_state['step_loss']) / len(self.active_state['step_loss'])
            mean_accuracy = sum(self.active_state['step_accuracy']) / len(self.active_state['step_accuracy'])

            # xi_est = (torch.stack(self.active_state['local_step_gradients']) - torch.stack(self.active_state['gradient_ref'])).abs().mean(dim=0).mean().item()
            sigma_est = np.mean(self.active_state['sigma_est'])
            xi_est = np.mean(self.active_state['xi_est'])

            self.active_state = None
            return {
                'delt_w': flat_delts, 
                # 'grad': flat_grad,
                'xi_est': xi_est,
                'sigma_est': sigma_est,
                'mean_loss': mean_loss,
                'mean_accuracy': mean_accuracy,
                'done': True,
            } 
        else:
            return {
                'done': False,
            }     


    def estimate_sigma_xi(self, mean_grad: torch.Tensor, ref_grad: torch.Tensor, input_batch: Tuple[torch.Tensor]) -> float:
        images, labels = input_batch
        batch_size = images.shape[0]
        sigma_local, xi_local = [], []
        selections = np.random.choice(batch_size, size=self.est_samples, replace=False)

        for idx in selections:
            image, label = images[idx], labels[idx]
            self.net.zero_grad()
            nn_output = self.net(image)
            if self.args.task == 'AutoEnc':
                loss = self.loss_func(nn_output, image) / image.shape[-1] / image.shape[-2] / image.shape[-3]
                nnout_max = None
            else:
                nnout_max = torch.argmax(nn_output, dim=1, keepdim=False)                
                loss = self.loss_func(nn_output, label) 
            loss.backward()
            flat_grad: torch.Tensor = gather_flat_grad(self.net)
            sigma_local.append((flat_grad - mean_grad).square())
            xi_local.append((flat_grad - ref_grad).square())

        sigma_est:float = torch.stack(sigma_local).mean(dim=0).sum().sqrt().item()
        sigma_est:float = sigma_est / np.sqrt(float(batch_size)) # Adjust for the fact that each data sample is noisier than a batch of data samples

        xi_est:float = torch.stack(xi_local).mean(dim=0).sum().sqrt().item()
        xi_est:float = xi_est / np.sqrt(float(batch_size)) # Adjust for the fact that each data sample is noisier than a batch of data samples

        return sigma_est, xi_est


    def sync_update(self, 
        net: Optional[NNet]=None, 
        gradient_ref: Optional[torch.Tensor]=None, 
    ) -> None:
        # Do not modify self.value if value is None
        if net is not None:
            self.net = net
        if gradient_ref is not None:
            self.gradient_ref = gradient_ref


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




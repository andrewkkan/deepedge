#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch

def do_MAS_Glob(args, local_user, net_glob, omega_sum, N_omega):
    if omega_sum and local_user.omega_local and N_omega > 0:
        # When the conditions are all met (1 and 2):
        # 1. This local user is not the first user globally to execute do_MAS_Glob
        # 2. This local user has visited do_MAS before and hence has a old set of omega_local values
        # Condition 2 implies 1, explicitly, but let's just check them all.
        for layer_idx in range(len(omega_sum)):
            omega_sum[layer_idx]['weight'] = omega_sum[layer_idx]['weight'] - local_user.omega_local[layer_idx]['weight']
            omega_sum[layer_idx]['bias'] = omega_sum[layer_idx]['bias'] - local_user.omega_local[layer_idx]['bias']
        N_omega = N_omega - 1
    if omega_sum == None:
        # When the condition is met such that this local user is the first user globally to execute do_MAS_Glob
        local_user.omega_update(net=copy.deepcopy(net_glob).to(args.device), omega_glob=[], N_omega=0)
        omega_sum = copy.deepcopy(local_user.omega_local)
    else:
        local_user.omega_update(net=copy.deepcopy(net_glob).to(args.device), omega_glob=copy.deepcopy(omega_sum), N_omega=N_omega)
        for layer_idx in range(len(omega_sum)):
            omega_sum[layer_idx]['weight'] = omega_sum[layer_idx]['weight'] + local_user.omega_local[layer_idx]['weight']
            omega_sum[layer_idx]['bias'] = omega_sum[layer_idx]['bias'] + local_user.omega_local[layer_idx]['bias']
    N_omega = N_omega + 1
    return omega_sum, N_omega


def do_Omega_Local_Update(local_user, net, omega_glob, N_omega):
    local_user.omega_global = omega_glob
    if N_omega > 0:
        for idx in range(len(omega_glob)):
            local_user.omega_global[idx]['weight'] = local_user.omega_global[idx]['weight'] / float(N_omega)
            local_user.omega_global[idx]['bias'] = local_user.omega_global[idx]['bias'] / float(N_omega)

    local_user.omega_local = []
    for idx, layer in enumerate(net.modules()):
        if type(layer) == torch.nn.modules.linear.Linear:
            local_user.omega_local.append({
                    'idx': idx,
                    'weight': torch.zeros(layer.weight.shape).to(local_user.args.device).to(local_user.args.device),
                    'bias': torch.zeros(layer.bias.shape).to(local_user.args.device).to(local_user.args.device),
            })
    sample_counter = 0
    for batch_idx, (images, labels) in enumerate(local_user.ldr_train):
        images, labels = images.to(local_user.args.device), labels.to(local_user.args.device)
        for idx in range(images.shape[0]):
            nn_outputs = net(images[idx])
            nnout_max = torch.argmax(nn_outputs, dim=1, keepdim=False)
            if nnout_max[0] == labels[idx]:
                backward_vector = torch.nn.functional.one_hot(nnout_max, nn_outputs.shape[1]).to(torch.float)
            else:
                continue
            net.zero_grad()
            nn_outputs.backward(backward_vector, retain_graph=True)
            net_layer_list = list(net.modules())
            for omega_layer in local_user.omega_local:
                omega_layer['weight'] = omega_layer['weight'] + torch.abs(net_layer_list[omega_layer['idx']].weight.grad.data).to(local_user.args.device)
                omega_layer['bias'] = omega_layer['bias'] + torch.abs(net_layer_list[omega_layer['idx']].bias.grad.data).to(local_user.args.device)
            sample_counter = sample_counter + 1
    if sample_counter:
        for omega_layer in local_user.omega_local:
            omega_layer['weight'] = omega_layer['weight'] / float(sample_counter)
            omega_layer['bias'] = omega_layer['bias'] / float(sample_counter)


def calculate_Regularization_Omega(net_glob, net_local, omega_glob):
    summation = 0.
    net_glob = list(net_glob.modules())
    net_local = list(net_local.modules())
    for layer_idx, omega_layer in enumerate(omega_glob):
        summation = summation + torch.sum(omega_layer['weight'] * torch.pow(net_glob[omega_layer['idx']].weight - net_local[omega_layer['idx']].weight, 2.))
        summation = summation + torch.sum(omega_layer['bias'] * torch.pow(net_glob[omega_layer['idx']].bias - net_local[omega_layer['idx']].bias, 2.))
    return summation
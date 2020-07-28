#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from IPython import embed

def FedAvg(w, local_size=None):
    if local_size:
        scale = []
        sum_size = 0
        for ls in local_size:
            sum_size += ls
        for ls in local_size:
            scale.append(float(ls) / sum_size)
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            w_avg[k] = w_avg[k] * scale[0] 
            for i in range(1, len(w)):
                w_avg[k] += w[i][k] * scale[i] 
    else:
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedAvgUpdate(wl_new, wl_old, w_global, scale=0.0):
    if scale == 0.0:
        scale = len(wl_new)
    w_avg = copy.deepcopy(w_global)
    for k in w_avg.keys():
        for i in range(len(wl_new)):
            w_avg[k] += (wl_new[i][k] - wl_old[i][k]) / scale
    return w_avg

def FedAvgMomentum(wl_new, wl_old, w_global, v_sd, momentum, epoch_idx, scale=None):
    if scale is None:
        scale = torch.ones(len(wl_new)) / float(len(wl_new))
    w_avg = copy.deepcopy(w_global)
    for k in w_avg.keys():
        vd = 0.
        for i in range(len(wl_new)):
            vd += (wl_new[i][k] - wl_old[i][k]) * scale[i]
        if epoch_idx == 0:
            v_sd[k] = vd
        else:
            v_sd[k] = momentum * v_sd[k] + (1. - momentum) * vd
        w_avg[k] += v_sd[k]
    return w_avg
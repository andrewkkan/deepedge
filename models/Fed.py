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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedGM(w, p):
    p = max(1., p)
    # Generalized mean with p = 1 is identical to arithmetic mean used in FedAvg
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(0, len(w)):
            if i == 0:
                w_avg[k] = torch.sign(w[i][k]) * torch.pow(torch.abs(w[i][k]), p) 
            else:
                w_avg[k] += torch.sign(w[i][k]) * torch.pow(torch.abs(w[i][k]), p) 
        w_avg[k] = torch.sign(w_avg[k]) * torch.pow(torch.abs(w_avg[k] / len(w) ), 1./p)
    return w_avg

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from __future__ import print_function

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from IPython import embed

def DFAN_ensemble(args, teacher, student, generator, optimizer, epoch):

    for ii in range(10):
        teacher[ii].eval()
    generator.train()
    student.train()
    loss_G = torch.tensor(0.0)
    loss_S = torch.tensor(0.0)
    optimizer_S, optimizer_G = optimizer
    sm = torch.nn.Softmax()

    for i in range(args.epoch_itrs):
        for j in range(5):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            optimizer_S.zero_grad()
            fake = generator(z).detach()
            fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
            s_logit = student(fake)
            t_logit = ensemble(teacher, fake, detach=True, mode=args.ensemble_mode)
            oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
            loss_S = torch.log(1. / (2. - oneMinus_P_S))
            loss_S.backward()
            optimizer_S.step()
        for k in range(1):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            optimizer_G.zero_grad()
            fake = generator(z)
            fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
            s_logit = student(fake)
            t_logit = ensemble(teacher, fake, detach=False, mode=args.ensemble_mode)
            oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
            max_Gout = torch.max(torch.abs(fake))
            if max_Gout > 8.0:
                loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0) + torch.pow(max_Gout - 8.0, 2.0)
                print(max_Gout)
            else:
                loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0)
            loss_G.backward()                   
            optimizer_G.step()

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))        #vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            #vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())


def ensemble(teacher, input, detach=True, mode=1):
    t_logit = []
    if detach:
        for k in range(10):
            t_logit.append(teacher[k](input).detach())
    else:
        for k in range(10):
            t_logit.append(teacher[k](input))
    if mode == 1:
        t_entropy_inv = []
        t_scale = []
        # Scaling by inverse of entropy
        for k in range(10):
            t_entropy_inv.append(1.0 / Categorical(probs = F.softmax(t_logit[k], dim=-1)).entropy())
        # t_scale_sum = torch.zeros_like(t_entropy_inv[0])
        # for k in range(10):
        #     t_scale_sum += t_entropy_inv[k]
        for k in range(10):
            t_scale.append(t_entropy_inv[k] / 1.0)
        t_logit_out = torch.zeros_like(t_logit[0])
        for k in range(10):
            t_logit_out += t_logit[k] * t_scale[k].reshape(-1,1).repeat(1,10)
    elif mode == 2:
        # Gating
        t_entropy = torch.zeros_like(t_logit[0])
        for k in range(10):
            t_entropy[:,k] = Categorical(probs = F.softmax(t_logit[k], dim=-1)).entropy()
        minval, argmin = t_entropy.min(1)
        minval_cond = minval.reshape(-1,1).repeat(1,10)
        min_mask = t_entropy == minval_cond
        min_mask = min_mask.float()
        t_logit_out = torch.zeros_like(t_logit[0])
        for k in range(10):
            t_logit_out += t_logit[k] * min_mask[:,k].reshape(-1,1).repeat(1,10)
    elif mode == 0:
        t_logit_out = torch.zeros_like(t_logit[0])
        for k in range(10):
            t_logit_out += t_logit[k] 

    return t_logit_out










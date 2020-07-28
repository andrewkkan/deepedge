#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from __future__ import print_function

import torch
import copy
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from torch import nn

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import norm

from models.Fed import FedAvg
from models.test import test_img, test_img_ensem
from models.Nets import MLP

from IPython import embed

def DFAN_ensemble(args, teacher, student, generator, optimizer, epoch):
    for local in teacher:
        local.eval()
    generator.train()
    student.train()
    loss_G = torch.tensor(0.0)
    loss_S = torch.tensor(0.0)
    optimizer_S, optimizer_G = optimizer
    sm = torch.nn.Softmax()

    loss_S1 = 0.0
    for i in range(args.epoch_itrs):
        if loss_S1 < -0.69:
            break
        if i > 0:
            for k in range(1):
                z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
                optimizer_G.zero_grad()
                fake = generator(z)
                fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
                s_logit = student(fake)
                # t_sm = ensemble(teacher, fake, detach=False, mode=args.ensemble_mode)
                t_logit = torch.zeros_like(teacher[0](fake))
                for t in teacher:
                    t_logit += t(fake)
                oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
                max_Gout = torch.max(torch.abs(fake))
                if max_Gout > 8.0:
                    loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0) + torch.pow(max_Gout - 8.0, 2.0)
                    print(max_Gout)
                else:
                    loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0)
                # loss_G += -(sm(t_logit) * sm(t_logit).log()).mean() * 0.001
                loss_G.backward()                   
                optimizer_G.step()
        for j in range(5):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            optimizer_S.zero_grad()
            fake = generator(z).detach()
            fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
            s_logit = student(fake)
            # t_sm = ensemble(teacher, fake, detach=True, mode=args.ensemble_mode)
            t_logit = torch.zeros_like(teacher[0](fake).detach())
            for t in teacher:
                t_logit += t(fake).detach()
            oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
            loss_S1 = torch.log(1. / (2. - oneMinus_P_S))
            loss_S2 = torch.zeros_like(loss_S1)
            # if w_reg and alpha > 0.0:
            #     for wr, ws in zip(w_reg.values(), student.parameters()):
            #         loss_S2 += (wr - ws).pow(2.).sum() * alpha
            loss_S = loss_S1 + loss_S2
            loss_S.backward()
            optimizer_S.step()

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))        #vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            #vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())

def ensemble(teacher, input, detach=True, mode=1):
    sm = torch.nn.Softmax()

    t_logit = []
    if detach:
        for local in teacher:
            t_logit.append(local(input).detach())
    else:
        for local in teacher:
            t_logit.append(local(input))
    if mode == 1:
        t_entropy_inv = []
        t_scale = []
        # Scaling by inverse of entropy
        for t_l in t_logit:
            t_entropy_inv.append(1.0 / Categorical(probs = F.softmax(t_l, dim=-1)).entropy())
        t_scale_sum = torch.zeros_like(t_entropy_inv[0])
        for t_ent_inv in t_entropy_inv:
            t_scale_sum += t_ent_inv
        for t_ent_inv in t_entropy_inv:
            t_scale.append(t_ent_inv / t_scale_sum)
        t_sm_out = torch.zeros_like(t_logit[0])
        for t_l, t_S in zip(t_logit, t_scale):
            t_sm_out += sm(t_l) * t_s.reshape(-1,1).repeat(1,len(teacher))
    elif mode == 0:
        t_sm_out = torch.zeros_like(t_logit[0])
        for t_l in t_logit:
            t_sm_out += sm(t_l) / float(len(teacher))

    return t_sm_out


def DFAN_multigen(args, teacher, gen_sd, ref_sd, student, proxy, generator, optimizer, epoch, dataset_test):

    for t in teacher:
        t.eval()
    generator.train()
    proxy.train()
    student.train()
    optimizer_P, optimizer_G = optimizer
    sm = torch.nn.Softmax()
    layers = ['layer_input', 'layer_hidden1', 'layer_hidden2']

    student.load_state_dict(ref_sd)
    acc_dfan_ensem, _ = test_img(student, dataset_test, args)
    temp_student = []
    for t in teacher:
        temp_student.append(copy.deepcopy(t))
    acc_ensem, _ = test_img_ensem(temp_student, dataset_test, args)
    print("-1 ", acc_dfan_ensem, acc_ensem)
    # mlpt = fusion_neuron_MLP_mg(teacher, ref_sd, 1024, 200, 10)
    # mlpt = mlpt.to(args.device)
    # mlpt.eval()
    # print(mlpt.alpha)
    # ref_sd = student.state_dict()
    proxy_sd, loss_S = [], []
    for ti in enumerate(teacher):
        # proxy_sd.append(copy.deepcopy(mlpt.state_dict()))
        proxy_sd.append(copy.deepcopy(ref_sd))
        # proxy_sd.append(copy.deepcopy(w_fedavg))
        loss_S.append(0.)

    w_fedavg_start = FedAvg(proxy_sd)
    sign_changes = torch.BoolTensor([False for idx in range(len(proxy_sd))])
    w_proxy_cossim_last = None
    for i in range(args.epoch_itrs):
        z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
        # if loss_S1[0] and np.sum(np.array(loss_S1) < -0.69) == 10:
        #     break
        if np.sum(np.array(loss_S) < -0.68) == len(teacher):
            break
        for ii in range(len(teacher)):
            if loss_S[ii] and loss_S[ii] < -0.68:
                continue
            generator.load_state_dict(gen_sd[ii])
            proxy.load_state_dict(proxy_sd[ii]) 
            for k in range(2):
                optimizer_G.zero_grad()
                fake = generator(z)
                fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
                s_logit = proxy(fake)
                t_logit = teacher[ii](fake)
                oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
                max_Gout = torch.max(torch.abs(fake))
                if max_Gout > 8.0:
                    loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0) + torch.pow(max_Gout - 8.0, 2.0)
                    print(max_Gout)
                else:
                    loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0)
                loss_G.backward()                   
                optimizer_G.step()
            for j in range(5):
                optimizer_P.zero_grad()
                fake = generator(z).detach()
                fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
                s_logit = proxy(fake)
                t_logit = teacher[ii](fake).detach()
                oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
                loss_S[ii] = torch.log(1. / (2. - oneMinus_P_S))
                loss_R = torch.FloatTensor([0.]).to(args.device)
                for ref_w, proxy_w in zip(ref_sd.values(), proxy.parameters()):
                    loss_R += (ref_w - proxy_w).pow(2.).sum() * args.alpha_scale
                loss_SR = loss_S[ii] + loss_R
                loss_SR.backward()
                optimizer_P.step()
                # print(i, ii, loss_S1.cpu().detach().numpy(), loss_S2.cpu().detach().numpy(), loss_S.cpu().detach().numpy())
            gen_sd[ii] = copy.deepcopy(generator.state_dict())
            proxy_sd[ii] = copy.deepcopy(proxy.state_dict())

        if args.cossim_filter:
            w_fedavg_optim = FedAvg(proxy_sd)
            w_fedavg_delta = {}
            for ln in layers:
                w_fedavg_delta[ln] = torch.cat((w_fedavg_optim[ln+'.weight'], w_fedavg_optim[ln+'.bias'].unsqueeze(1)), dim=1) - torch.cat((w_fedavg_start[ln+'.weight'], w_fedavg_start[ln+'.bias'].unsqueeze(1)), dim=1)
            w_proxy_cossim = torch.zeros(len(teacher))
            for ii in range(len(teacher)):
                for ln in layers:
                    w_proxy_delta = torch.cat((proxy_sd[ii][ln+'.weight'], proxy_sd[ii][ln+'.bias'].unsqueeze(1)), dim=1) - torch.cat((w_fedavg_optim[ln+'.weight'], w_fedavg_optim[ln+'.bias'].unsqueeze(1)), dim=1)
                    w_proxy_cossim[ii] += (w_fedavg_delta[ln] * w_proxy_delta).sum() / w_fedavg_delta[ln].norm(2) / w_proxy_delta.norm(2) / float(len(layers))
            if w_proxy_cossim_last != None:
                sign_changes |= ((w_proxy_cossim * w_proxy_cossim_last).sign() < 0)
            w_proxy_cossim_last = copy.deepcopy(w_proxy_cossim)

            print(sign_changes, w_proxy_cossim)

            if sign_changes.sum() >= 5:
                cossimfilt_proxy_sd = []
                for pi, psd in enumerate(proxy_sd):
                    if sign_changes[pi] == True or w_proxy_cossim[pi].abs() < 0.1:
                        cossimfilt_proxy_sd.append(proxy_sd[pi]) 
                student.load_state_dict(FedAvg(cossimfilt_proxy_sd))
                break

        # if epoch > -1:
        #     acc_glob, _ = test_img(student, dataset_test, args)
        #     temp_student = []
        #     for ri in range(10):
        #         temp_student.append(MLP(1024, 200, 10).to(args.device))
        #         temp_student[ri].load_state_dict(proxy_sd[ri])
        #     acc_ensem, _ = test_img_ensem(temp_student, dataset_test, args)
        #     print(i, acc_glob, acc_ensem)
        # if (i%4) == 1:
        #     ref_sd = w_fedavg
        #     for ii in range(10):
        #         proxy_sd[ii] = copy.deepcopy(ref_sd)
    w_fedavg = FedAvg(proxy_sd)
    student.load_state_dict(w_fedavg)
    return proxy_sd


def DFAN_single(args, teacher, student, generator, optimizer, epoch, reg=False):

    teacher.eval()
    generator.train()
    student.train()
    loss_G = torch.tensor(0.0).to(args.device)
    loss_S1 = torch.tensor(0.0).to(args.device)
    loss_S2 = torch.tensor(0.0).to(args.device)
    optimizer_S, optimizer_G = optimizer
    sm = torch.nn.Softmax()

    ref_sd = copy.deepcopy(student.state_dict())
    for i in range(args.epoch_itrs):
        if loss_S1 < -0.69:
            break
        if i > 0:
            for k in range(1):
                z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
                optimizer_G.zero_grad()
                fake = generator(z)
                fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
                s_logit = student(fake)
                # t_sm = ensemble(teacher, fake, detach=False, mode=args.ensemble_mode)
                t_logit = teacher(fake)
                oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
                max_Gout = torch.max(torch.abs(fake))
                if max_Gout > 8.0:
                    loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0) + torch.pow(max_Gout - 8.0, 2.0)
                    print(max_Gout)
                else:
                    loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0)
                loss_G.backward()                   
                optimizer_G.step()
        for j in range(5):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            optimizer_S.zero_grad()
            fake = generator(z).detach()
            fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
            s_logit = student(fake)
            # t_sm = ensemble(teacher, fake, detach=True, mode=args.ensemble_mode)
            t_logit = teacher(fake).detach()
            oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
            loss_S1 = torch.log(1. / (2. - oneMinus_P_S))
            if reg:
                loss_S2 = torch.FloatTensor([0.]).to(args.device)
                for ref_w, student_w in zip(ref_sd.values(), student.parameters()):
                    loss_S2 += (ref_w - student_w).pow(2.).sum() * args.alpha_scale
            loss_S = loss_S1 + loss_S2
            loss_S.backward()
            optimizer_S.step()

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))        #vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            #vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())


def DFAN_regavg(args, teacher, student, generator, optimizer, epoch, dataset_test):

    for local in teacher:
        local.eval()
    generator.train()
    student.train()
    loss_G = torch.tensor(0.0)
    loss_S = torch.tensor(0.0)
    optimizer_S, optimizer_G = optimizer
    sm = torch.nn.Softmax()

    # w_locals = []
    # for local in teacher:
    #     w_locals.append(copy.deepcopy(local.state_dict()))
    # w_fedavg = FedAvg(w_locals)

    # mlpa, mlpt = model_fusion_MLP(teacher, 1024, 200, 10)
    # mlpa, mlpt = mlpa.to(args.device), mlpt.to(args.device)
    # w_mlpt = mlpt.state_dict()
    mlpa, mlpt = fusion_layer_MLP(args, teacher, student, args.img_size[0]*args.img_size[1]*args.img_size[2], 200, args.num_classes)
    mlpa = mlpa.to(args.device)
    mlpt = mlpt.to(args.device)
    mlpa.eval()
    mlpt.eval()

    w_fedavg = mlpt.state_dict()

    # acc_mlpa, _ = test_img(mlpa, dataset_test, args, shuffle=True)
    # acc_mlpt, _ = test_img(mlpt, dataset_test, args, shuffle=True)
    # print("MLPA accuracy = ", acc_mlpa)
    # print("MLPT accuracy = ", acc_mlpt)
    for i in range(args.epoch_itrs):
        # acc_glob, _ = test_img(student, dataset_test, args, shuffle=True)
        # print("Net_glob accuracy = ", acc_glob)
        for k in range(2):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            optimizer_G.zero_grad()
            fake = generator(z)
            fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
            s_logit = student(fake)
            # t_sm = ensemble(teacher, fake, detach=False, mode=args.ensemble_mode)
            # oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), t_sm))
            # t_logit = torch.zeros_like(teacher[0](fake))
            # for k in range(10):
            #     t_logit += teacher[k](fake) 
            t_logit = mlpa(fake)
            oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
            max_Gout = torch.max(torch.abs(fake))
            if max_Gout > 8.0:
                loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0) + torch.pow(max_Gout - 8.0, 2.0)
                print(max_Gout)
            else:
                loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0)
            loss_G.backward()                   
            optimizer_G.step()
        for j in range(5):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            optimizer_S.zero_grad()
            fake = generator(z).detach()
            fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
            s_logit = student(fake)
            # t_sm = ensemble(teacher, fake, detach=True, mode=args.ensemble_mode)
            # # oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), t_sm))
            # t_logit = torch.zeros_like(teacher[0](fake).detach())
            # for k in range(10):
            #     t_logit += teacher[k](fake).detach()
            t_logit = mlpa(fake).detach()
            oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
            loss_S1 = torch.log(1. / (2. - oneMinus_P_S))

            diff_L2 = torch.FloatTensor([0.]).to(args.device)
            # for ln, student_w, fedavg_w in zip(student.state_dict().keys(), student.parameters(), w_fedavg.values()):
            #     diff_L2 += ((fedavg_w - student_w) * mlpt.alpha[ln].to(args.device)).norm(2) * args.alpha_scale
            for ln, student_w, mlpt_w in zip(student.state_dict().keys(), student.parameters(), mlpt.parameters()):
                diff_L2 += ((mlpt_w - student_w)*mlpt.alpha[ln]).norm(2)
            if epoch == 0:
                loss_S2 = 0
            else:
                loss_S2 = diff_L2

            # diff_L2 = torch.FloatTensor([0.]).to(args.device)
            # for student_w, mlpt_w in zip(student.parameters(), w_mlpt.values()):
            #     diff_L2 += ((mlpt_w - student_w)*mlpt_w.abs()).norm(2)
            # loss_S2 = diff_L2 * alpha

            # print(loss_S1, loss_S2)

            loss_S = loss_S1 + loss_S2
            loss_S.backward()
            optimizer_S.step()

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))        #vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            #vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())

        if loss_S1 < -0.69:
            break


def fusion_layer_MLP(args, models, ref, dim_in, dim_hidden, dim_out):
    # eps= 0.25
    eps = args.fuse_eps
    num_models = len(models)

    layers = ['layer_input', 'layer_hidden1', 'layer_hidden2']
    s_sd = ref.state_dict()
    delta = {}
    for lni, ln in enumerate(layers):
        delta[ln] = torch.zeros(len(models), s_sd[ln+'.weight'].shape[0] * (s_sd[ln+'.weight'].shape[1]+1))
        for ii, t in enumerate(models):
            t_sd = t.state_dict()
            delta[ln][ii, 0:(s_sd[ln+'.weight'].shape[0]*s_sd[ln+'.weight'].shape[1])] = (t_sd[ln+'.weight'] - s_sd[ln+'.weight']).flatten()
            delta[ln][ii, (s_sd[ln+'.weight'].shape[0]*s_sd[ln+'.weight'].shape[1]):s_sd[ln+'.weight'].shape[0] * (s_sd[ln+'.weight'].shape[1]+1)] = t_sd[ln+'.bias'] - s_sd[ln+'.bias']

    cluster_labels = {
        'layer_input': np.zeros(num_models),
        'layer_hidden1': np.zeros(num_models),
        'layer_hidden2': np.zeros(num_models),
    }
    for ln, dt in delta.items():
        db_agg = DBSCAN(eps=eps, min_samples=1).fit(dt)
        cluster_labels[ln] = db_agg.labels_

    return MLP_agg_layer(dim_in, dim_hidden, dim_out, models, cluster_labels), MLP_trimmed_layer(dim_in, dim_hidden, dim_out, models, cluster_labels)


class MLP_agg_layer(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, models, cluster_labels):
        super().__init__()
        num_models = len(models)
        num_clusters, num_neurons = {},{}
        num_neurons_original,                   num_synapses_original                   =  {},              {}
        num_neurons_original['layer_input'],    num_synapses_original['layer_input']    = int(dim_hidden),  int(dim_in)
        num_neurons_original['layer_hidden1'],  num_synapses_original['layer_hidden1']  = int(dim_hidden),  int(dim_hidden)
        num_neurons_original['layer_hidden2'],  num_synapses_original['layer_hidden2']  = int(dim_out),     int(dim_hidden)     
        for ln, cl in cluster_labels.items():
            num_clusters[ln] = cl.max() + 1
            num_neurons[ln] = int(num_clusters[ln] * num_neurons_original[ln])
        num_neurons['branch_in'] = num_models*dim_in
        num_neurons['ensemble_out'] = dim_out
        self.branch_in = nn.Linear(dim_in, num_models*dim_in)
        self.layer_input = nn.Linear(dim_in*num_models, num_neurons['layer_input'])
        self.layer_hidden1 = nn.Linear(num_neurons['layer_input'], num_neurons['layer_hidden1'])
        self.layer_hidden2 = nn.Linear(num_neurons['layer_hidden1'], num_neurons['layer_hidden2'])
        self.relu = nn.ReLU()
        self.ensemble_out = nn.Linear(num_neurons['layer_hidden2'], dim_out)
        neurons_ensemble = {
            'branch_in.weight': torch.zeros(num_models*dim_in, dim_in),
            'branch_in.bias': torch.zeros(num_models*dim_in),
            'layer_input.weight': torch.zeros(num_models*dim_hidden, num_models*dim_in),
            'layer_input.bias': torch.zeros(num_models*dim_hidden),
            'layer_hidden1.weight': torch.zeros(num_models*dim_hidden, num_models*dim_hidden),
            'layer_hidden1.bias': torch.zeros(num_models*dim_hidden),
            'layer_hidden2.weight': torch.zeros(num_models*dim_out, num_models*dim_hidden),
            'layer_hidden2.bias': torch.zeros(num_models*dim_out),
            'ensemble_out.weight': torch.zeros(dim_out, num_models*dim_out),
            'ensemble_out.bias': torch.zeros(dim_out),
        }
        for mi, md in enumerate(models):
            msd = md.state_dict()
            neurons_ensemble['branch_in.weight'][mi*dim_in:(mi+1)*dim_in, :] = torch.eye(dim_in)
            neurons_ensemble['layer_input.weight'][mi*dim_hidden:(mi+1)*dim_hidden, mi*dim_in:(mi+1)*dim_in] = msd['layer_input.weight']
            neurons_ensemble['layer_input.bias'][mi*dim_hidden:(mi+1)*dim_hidden] = msd['layer_input.bias']
            neurons_ensemble['layer_hidden1.weight'][mi*dim_hidden:(mi+1)*dim_hidden, mi*dim_hidden:(mi+1)*dim_hidden] = msd['layer_hidden1.weight']
            neurons_ensemble['layer_hidden1.bias'][mi*dim_hidden:(mi+1)*dim_hidden] = msd['layer_hidden1.bias']
            neurons_ensemble['layer_hidden2.weight'][mi*dim_out:(mi+1)*dim_out, mi*dim_hidden:(mi+1)*dim_hidden] = msd['layer_hidden2.weight']
            neurons_ensemble['layer_hidden2.bias'][mi*dim_out:(mi+1)*dim_out] = msd['layer_hidden2.bias']
            neurons_ensemble['ensemble_out.weight'][:, mi*dim_out:(mi+1)*dim_out] = torch.eye(dim_out)
        # Row reduction with weight averaging   
        # This is the neuron fusing portion    
        neurons_row_reduced = {
            'branch_in.weight': neurons_ensemble['branch_in.weight'],
            'branch_in.bias': neurons_ensemble['branch_in.bias'],
            'layer_input.weight': torch.zeros(num_neurons['layer_input'], num_models*dim_in),
            'layer_input.bias': torch.zeros(num_neurons['layer_input']),
            'layer_hidden1.weight': torch.zeros(num_neurons['layer_hidden1'], num_models*dim_hidden),
            'layer_hidden1.bias': torch.zeros(num_neurons['layer_hidden1']),
            'layer_hidden2.weight': torch.zeros(num_neurons['layer_hidden2'], num_models*dim_hidden),
            'layer_hidden2.bias': torch.zeros(num_neurons['layer_hidden2']),
            'ensemble_out.weight': neurons_ensemble['ensemble_out.weight'],
            'ensemble_out.bias': neurons_ensemble['ensemble_out.bias'],
        }
        cluster_to_model_mappings = {
                                            'layer_input': [],
                                            'layer_hidden1': [],
                                            'layer_hidden2': [],
        }
        for ln, cl in cluster_labels.items():
            for ci in range(int(num_clusters[ln])): # ci is cluster index.  neuron_cluster_to_model_mappings[ln][ni][ci] stores the model indexes for each cluster per neuron
                cluster_to_model_mappings[ln].append(np.argwhere(cl == ci).flatten()) # finds the model index that matches the cluster index
        for ln, cmap_ln in cluster_to_model_mappings.items():
            row_index = 0
            for ci, cmap_ci in enumerate(cmap_ln):
                for ni in range(num_neurons_original[ln]):
                    neuron_indices_to_average = cmap_ci*num_neurons_original[ln]+ni
                    weights_to_average = torch.zeros(len(cmap_ci), num_synapses_original[ln])
                    for nita_idx, nita in enumerate(neuron_indices_to_average):
                        weights_to_average[nita_idx, :] = neurons_ensemble[ln+'.weight'][nita][cmap_ci[nita_idx]*num_synapses_original[ln]:(cmap_ci[nita_idx]+1)*num_synapses_original[ln]]
                    weights_averaged = weights_to_average.mean(dim=0)
                    for nita_idx, nita in enumerate(neuron_indices_to_average):
                        if nita_idx == 0:
                            neurons_row_reduced[ln+'.weight'][row_index][cmap_ci[nita_idx]*num_synapses_original[ln]:(cmap_ci[nita_idx]+1)*num_synapses_original[ln]] = weights_averaged
                        else:
                            neurons_row_reduced[ln+'.weight'][row_index][cmap_ci[nita_idx]*num_synapses_original[ln]:(cmap_ci[nita_idx]+1)*num_synapses_original[ln]] = torch.zeros_like(weights_averaged)
                    neurons_row_reduced[ln+'.bias'][row_index] = neurons_ensemble[ln+'.bias'][cmap_ci*num_neurons_original[ln]+ni].mean(dim=0)
                    row_index += 1

        # Column reduction with weight averaging
        # This is the portion to tidy-up next-layer (at L+1) synapses leading from a fused neuron (from layer L)
        # Weight averaging is needed when these next-layer (L+1) synapses leading from a fused neuron (from layer L) are part of the same neuron (from layer L+1)
        neurons_column_reduced = {
            'branch_in.weight': neurons_row_reduced['branch_in.weight'],
            'branch_in.bias': neurons_row_reduced['branch_in.bias'],
            'layer_input.weight': neurons_row_reduced['layer_input.weight'],
            'layer_input.bias': neurons_row_reduced['layer_input.bias'],
            'layer_hidden1.weight': torch.zeros(num_neurons['layer_hidden1'], num_neurons['layer_input']),
            'layer_hidden1.bias': neurons_row_reduced['layer_hidden1.bias'],
            'layer_hidden2.weight': torch.zeros(num_neurons['layer_hidden2'], num_neurons['layer_hidden1']),
            'layer_hidden2.bias': neurons_row_reduced['layer_hidden2.bias'],
            'ensemble_out.weight': torch.zeros(dim_out, num_neurons['layer_hidden2']),
            'ensemble_out.bias': neurons_row_reduced['ensemble_out.bias'],
        }
        next_layer = {
            'layer_input': 'layer_hidden1',
            'layer_hidden1': 'layer_hidden2',
            'layer_hidden2': 'ensemble_out',
        }
        for ln, cmap_ln in cluster_to_model_mappings.items():
            column_index = 0
            nln = next_layer[ln]
            for ci, cmap_ci in enumerate(cmap_ln):
                for ni in range(num_neurons_original[ln]):
                    synapse_indices_to_combine = cmap_ci*num_neurons_original[ln]+ni 
                    weights_to_combine = torch.zeros(num_neurons[nln], len(cmap_ci))
                    for sitc_idx, sitc in enumerate(synapse_indices_to_combine):
                        weights_to_combine[:, sitc_idx] = neurons_row_reduced[nln+'.weight'][:, sitc]
                    normalize_factors = (weights_to_combine != 0.0).double().sum(dim=1)
                    normalize_factors[normalize_factors == 0.] = torch.ones_like(normalize_factors[normalize_factors == 0.])
                    weights_combined = weights_to_combine.sum(dim=1) / normalize_factors
                    for sitc_idx, sitc in enumerate(synapse_indices_to_combine):
                        neurons_column_reduced[nln+'.weight'][:,column_index] = weights_combined
                    column_index += 1

        self.branch_in.load_state_dict({
            'weight':   neurons_column_reduced['branch_in.weight'],
            'bias':     neurons_column_reduced['branch_in.bias'],
        })
        self.layer_input.load_state_dict({
            'weight':   neurons_column_reduced['layer_input.weight'],
            'bias':     neurons_column_reduced['layer_input.bias'],
        })
        self.layer_hidden1.load_state_dict({
            'weight':   neurons_column_reduced['layer_hidden1.weight'],
            'bias':     neurons_column_reduced['layer_hidden1.bias'],
        })
        self.layer_hidden2.load_state_dict({
            'weight':   neurons_column_reduced['layer_hidden2.weight'],
            'bias':     neurons_column_reduced['layer_hidden2.bias'],
        })            
        self.ensemble_out.load_state_dict({
            'weight':   neurons_column_reduced['ensemble_out.weight'],
            'bias':     neurons_column_reduced['ensemble_out.bias'],
        })                           

    def forward(self, x):
        x = x.view(-1, x.shape[-3]*x.shape[-2]*x.shape[-1])
        x = self.branch_in(x)
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
        x = self.ensemble_out(x)
        return x

class MLP_trimmed_layer(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, models, cluster_labels):
        super().__init__()

        neurons_fedavg = {
            'layer_input.weight':       torch.zeros(dim_hidden, dim_in),
            'layer_input.bias':         torch.zeros(dim_hidden),
            'layer_hidden1.weight':     torch.zeros(dim_hidden, dim_hidden),
            'layer_hidden1.bias':       torch.zeros(dim_hidden),
            'layer_hidden2.weight':     torch.zeros(dim_out, dim_hidden),
            'layer_hidden2.bias':       torch.zeros(dim_out),
        }

        labels_cat = np.concatenate((cluster_labels['layer_input'], cluster_labels['layer_hidden1'], cluster_labels['layer_hidden2']), axis=0)
        max_cluster_label = None
        max_cluster_size = 0
        for l in range(int(np.max(labels_cat)+1)):
            size_for_label = len(labels_cat == l)
            if size_for_label > max_cluster_size:
                max_cluster_size = size_for_label
                max_cluster_label = l

        self.alpha = {}
        for ln in ['layer_input', 'layer_hidden1', 'layer_hidden2']:
            if max_cluster_size == 1:
                self.alpha[ln+'.weight'] = 0.0
            else:
                self.alpha[ln+'.weight'] = float((cluster_labels[ln] == max_cluster_label).sum()-1) / (len(models)-1)
            self.alpha[ln+'.bias'] = self.alpha[ln+'.weight']

        for ln, cl in cluster_labels.items():
            largest_cluster_idx = np.argwhere(cl == max_cluster_label).flatten()
            neuron_cluster = torch.zeros(len(largest_cluster_idx), neurons_fedavg[ln+'.weight'].shape[0], neurons_fedavg[ln+'.weight'].shape[1])
            neuron_bias = torch.zeros(len(largest_cluster_idx), neurons_fedavg[ln+'.weight'].shape[0])
            for lcii, lci in enumerate(largest_cluster_idx):
                neuron_cluster[lcii, :, :] = models[lci].state_dict()[ln+'.weight']
                neuron_bias[lcii, :] = models[lci].state_dict()[ln+'.bias']
            neurons_fedavg[ln+'.weight'] = neuron_cluster.mean(dim=0)
            neurons_fedavg[ln+'.bias'] = neuron_bias.mean(dim=0)

        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden2 = nn.Linear(dim_hidden, dim_out)
        self.relu = nn.ReLU()

        self.layer_input.load_state_dict({
            'weight':   neurons_fedavg['layer_input.weight'],
            'bias':   neurons_fedavg['layer_input.bias'],
        })
        self.layer_hidden1.load_state_dict({
            'weight':   neurons_fedavg['layer_hidden1.weight'],
            'bias':     neurons_fedavg['layer_hidden1.bias'],
        })
        self.layer_hidden2.load_state_dict({
            'weight':   neurons_fedavg['layer_hidden2.weight'],
            'bias':     neurons_fedavg['layer_hidden2.bias'],
        })                                     

    def forward(self, x):
        x = x.view(-1, x.shape[-3]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
        return x



def fusion_neuron_MLP_mg(args, models, s_sd, dim_in, dim_hidden, dim_out):
    # eps = 0.25
    eps = args.fuse_eps
    neurons_delta = {
        'layer_input': torch.zeros(dim_hidden, len(models), dim_in+1),
        'layer_hidden1': torch.zeros(dim_hidden, len(models), dim_hidden+1),
        'layer_hidden2': torch.zeros(dim_out, len(models), dim_hidden+1),
    }
    for mi, mnn in enumerate(models):
        # s_sd = ref.state_dict()
        for ln in neurons_delta.keys():
            neurons_delta[ln][:, mi, :] = torch.cat([
                (mnn.state_dict()[ln+'.weight'] - s_sd[ln+'.weight']), 
                (mnn.state_dict()[ln+'.bias'] - s_sd[ln+'.bias']).unsqueeze(dim=1),
            ], dim=1)
    cluster_labels = {
        'layer_input': np.zeros((dim_hidden, len(models))),
        'layer_hidden1': np.zeros((dim_hidden, len(models))),
        'layer_hidden2': np.zeros((dim_out, len(models))),
    }
    for ln, nd in neurons_delta.items():
        for n_idx in range(nd.shape[0]):
            db = DBSCAN(eps=eps, min_samples=1).fit(nd[n_idx,:,:])
            cluster_labels[ln][n_idx,:] = db.labels_

    return MLP_trimmed_neuron(dim_in, dim_hidden, dim_out, models, cluster_labels)


def fusion_neuron_MLP(models, ref, dim_in, dim_hidden, dim_out):
    eps = 0.5
    s_sd = ref.state_dict()
    neurons_delta = {
        'layer_input': torch.zeros(dim_hidden, len(models), dim_in+1),
        'layer_hidden1': torch.zeros(dim_hidden, len(models), dim_hidden+1),
        'layer_hidden2': torch.zeros(dim_out, len(models), dim_hidden+1),
    }
    for mi, mnn in enumerate(models):
        for ln in neurons_delta.keys():
            neurons_delta[ln][:, mi, :] = torch.cat([
                (mnn.state_dict()[ln+'.weight'] - s_sd[ln+'.weight']), 
                (mnn.state_dict()[ln+'.bias'] - s_sd[ln+'.bias']).unsqueeze(dim=1),
            ], dim=1)
    cluster_labels = {
        'layer_input': np.zeros((dim_hidden, len(models))),
        'layer_hidden1': np.zeros((dim_hidden, len(models))),
        'layer_hidden2': np.zeros((dim_out, len(models))),
    }
    for ln, nd in neurons_delta.items():
        for n_idx in range(nd.shape[0]):
            db = DBSCAN(eps=eps, min_samples=1).fit(nd[n_idx,:,:])
            cluster_labels[ln][n_idx,:] = db.labels_

    return MLP_agg_neuron(dim_in, dim_hidden, dim_out, models, cluster_labels), MLP_trimmed_neuron(dim_in, dim_hidden, dim_out, models, cluster_labels)
    # return MLP_trimmed_neuron(dim_in, dim_hidden, dim_out, models, cluster_labels)

class MLP_agg_neuron(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, models, cluster_labels):
        super().__init__()
        num_models = len(models)
        num_clusters, num_neurons = {},{}
        self.minavg_cluster_ratio = 1.0
        self.avg_cluster_ratio = 0.0
        for ln, cl in cluster_labels.items():
            num_clusters[ln] = cl.max(axis=1) + 1
            num_neurons[ln] = num_clusters[ln].sum().astype(np.int)
            self.minavg_cluster_ratio = min((1.0 - (float(num_neurons[ln]) / cluster_labels[ln].shape[0] / num_models)), self.minavg_cluster_ratio)
            self.avg_cluster_ratio += float(num_neurons[ln]) / cluster_labels[ln].shape[0] / num_models / len(cluster_labels.keys())
        num_neurons['branch_in'] = num_models*dim_in
        num_neurons['ensemble_out'] = dim_out
        self.branch_in = nn.Linear(dim_in, num_models*dim_in)
        self.layer_input = nn.Linear(dim_in*num_models, num_neurons['layer_input'])
        self.layer_hidden1 = nn.Linear(num_neurons['layer_input'], num_neurons['layer_hidden1'])
        self.layer_hidden2 = nn.Linear(num_neurons['layer_hidden1'], num_neurons['layer_hidden2'])
        self.relu = nn.ReLU()
        self.ensemble_out = nn.Linear(num_neurons['layer_hidden2'], dim_out)
        neurons_ensemble = {
            'branch_in.weight': torch.zeros(num_models*dim_in, dim_in),
            'branch_in.bias': torch.zeros(num_models*dim_in),
            'layer_input.weight': torch.zeros(num_models*dim_hidden, num_models*dim_in),
            'layer_input.bias': torch.zeros(num_models*dim_hidden),
            'layer_hidden1.weight': torch.zeros(num_models*dim_hidden, num_models*dim_hidden),
            'layer_hidden1.bias': torch.zeros(num_models*dim_hidden),
            'layer_hidden2.weight': torch.zeros(num_models*dim_out, num_models*dim_hidden),
            'layer_hidden2.bias': torch.zeros(num_models*dim_out),
            'ensemble_out.weight': torch.zeros(dim_out, num_models*dim_out),
            'ensemble_out.bias': torch.zeros(dim_out),
        }
        for mi, md in enumerate(models):
            msd = md.state_dict()
            neurons_ensemble['branch_in.weight'][mi*dim_in:(mi+1)*dim_in, :] = torch.eye(dim_in)
            neurons_ensemble['layer_input.weight'][mi*dim_hidden:(mi+1)*dim_hidden, mi*dim_in:(mi+1)*dim_in] = msd['layer_input.weight']
            neurons_ensemble['layer_input.bias'][mi*dim_hidden:(mi+1)*dim_hidden] = msd['layer_input.bias']
            neurons_ensemble['layer_hidden1.weight'][mi*dim_hidden:(mi+1)*dim_hidden, mi*dim_hidden:(mi+1)*dim_hidden] = msd['layer_hidden1.weight']
            neurons_ensemble['layer_hidden1.bias'][mi*dim_hidden:(mi+1)*dim_hidden] = msd['layer_hidden1.bias']
            neurons_ensemble['layer_hidden2.weight'][mi*dim_out:(mi+1)*dim_out, mi*dim_hidden:(mi+1)*dim_hidden] = msd['layer_hidden2.weight']
            neurons_ensemble['layer_hidden2.bias'][mi*dim_out:(mi+1)*dim_out] = msd['layer_hidden2.bias']
            neurons_ensemble['ensemble_out.weight'][:, mi*dim_out:(mi+1)*dim_out] = torch.eye(dim_out)
        neuron_cluster_to_model_mappings = {
            'layer_input': [],
            'layer_hidden1': [],
            'layer_hidden2': [],
        }
        for ln, cl in cluster_labels.items():
            for ni, nc in enumerate(num_clusters[ln]): # ni indexes neurons per layer, i.e. index of cluster_labels[ln].shape[0]
                neuron_cluster_to_model_mappings[ln].append([])
                for ci in range(int(nc)): # ci is cluster index.  neuron_cluster_to_model_mappings[ln][ni][ci] stores the model indexes for each cluster per neuron
                    neuron_cluster_to_model_mappings[ln][ni].append(np.argwhere(cl[ni, :] == ci).flatten()) # finds the model index that matches the cluster index
        num_neurons_original,                   num_synapses_original                   =  {},              {}
        num_neurons_original['layer_input'],    num_synapses_original['layer_input']    = int(dim_hidden),  int(dim_in)
        num_neurons_original['layer_hidden1'],  num_synapses_original['layer_hidden1']  = int(dim_hidden),  int(dim_hidden)
        num_neurons_original['layer_hidden2'],  num_synapses_original['layer_hidden2']  = int(dim_out),     int(dim_hidden)     
        # Row reduction with weight averaging   
        # This is the neuron fusing portion    
        neurons_row_reduced = {
            'branch_in.weight': neurons_ensemble['branch_in.weight'],
            'branch_in.bias': neurons_ensemble['branch_in.bias'],
            'layer_input.weight': torch.zeros(num_neurons['layer_input'], num_models*dim_in),
            'layer_input.bias': torch.zeros(num_neurons['layer_input']),
            'layer_hidden1.weight': torch.zeros(num_neurons['layer_hidden1'], num_models*dim_hidden),
            'layer_hidden1.bias': torch.zeros(num_neurons['layer_hidden1']),
            'layer_hidden2.weight': torch.zeros(num_neurons['layer_hidden2'], num_models*dim_hidden),
            'layer_hidden2.bias': torch.zeros(num_neurons['layer_hidden2']),
            'ensemble_out.weight': neurons_ensemble['ensemble_out.weight'],
            'ensemble_out.bias': neurons_ensemble['ensemble_out.bias'],
        }
        for ln, cmap_ln in neuron_cluster_to_model_mappings.items():
            row_index = 0
            for ni, cmap_ni in enumerate(cmap_ln): # ni indexes neurons per layer, i.e. index of cluster_labels[ln].shape[0]
                for ci, cmap_ci in enumerate(cmap_ni):
                    neuron_indices_to_average = cmap_ci*num_neurons_original[ln]+ni
                    weights_to_average = torch.zeros(len(cmap_ci), num_synapses_original[ln])
                    for nita_idx, nita in enumerate(neuron_indices_to_average):
                        weights_to_average[nita_idx, :] = neurons_ensemble[ln+'.weight'][nita][cmap_ci[nita_idx]*num_synapses_original[ln]:(cmap_ci[nita_idx]+1)*num_synapses_original[ln]]
                    weights_averaged = weights_to_average.mean(dim=0)
                    for nita_idx, nita in enumerate(neuron_indices_to_average):
                        if nita_idx == 0:
                            neurons_row_reduced[ln+'.weight'][row_index][cmap_ci[nita_idx]*num_synapses_original[ln]:(cmap_ci[nita_idx]+1)*num_synapses_original[ln]] = weights_averaged
                        else:
                            neurons_row_reduced[ln+'.weight'][row_index][cmap_ci[nita_idx]*num_synapses_original[ln]:(cmap_ci[nita_idx]+1)*num_synapses_original[ln]] = torch.zeros_like(weights_averaged)
                    neurons_row_reduced[ln+'.bias'][row_index] = neurons_ensemble[ln+'.bias'][cmap_ci*num_neurons_original[ln]+ni].mean(dim=0)
                    row_index += 1
        # Column reduction with weight averaging
        # This is the portion to tidy-up next-layer (at L+1) synapses leading from a fused neuron (from layer L)
        # Weight averaging is needed when these next-layer (L+1) synapses leading from a fused neuron (from layer L) are part of the same neuron (from layer L+1)
        neurons_column_reduced = {
            'branch_in.weight': neurons_row_reduced['branch_in.weight'],
            'branch_in.bias': neurons_row_reduced['branch_in.bias'],
            'layer_input.weight': neurons_row_reduced['layer_input.weight'],
            'layer_input.bias': neurons_row_reduced['layer_input.bias'],
            'layer_hidden1.weight': torch.zeros(num_neurons['layer_hidden1'], num_neurons['layer_input']),
            'layer_hidden1.bias': neurons_row_reduced['layer_hidden1.bias'],
            'layer_hidden2.weight': torch.zeros(num_neurons['layer_hidden2'], num_neurons['layer_hidden1']),
            'layer_hidden2.bias': neurons_row_reduced['layer_hidden2.bias'],
            'ensemble_out.weight': torch.zeros(dim_out, num_neurons['layer_hidden2']),
            'ensemble_out.bias': neurons_row_reduced['ensemble_out.bias'],
        }
        next_layer = {
            'layer_input': 'layer_hidden1',
            'layer_hidden1': 'layer_hidden2',
            'layer_hidden2': 'ensemble_out',
        }
        for ln, cmap_ln in neuron_cluster_to_model_mappings.items():
            column_index = 0
            nln = next_layer[ln]
            for ni, cmap_ni in enumerate(cmap_ln): # ni indexes neurons per layer, i.e. index of cluster_labels[ln].shape[0]
                for ci, cmap_ci in enumerate(cmap_ni):
                    synapse_indices_to_combine = cmap_ci*num_neurons_original[ln]+ni 
                    weights_to_combine = torch.zeros(num_neurons[nln], len(cmap_ci))
                    for sitc_idx, sitc in enumerate(synapse_indices_to_combine):
                        weights_to_combine[:, sitc_idx] = neurons_row_reduced[nln+'.weight'][:, sitc]
                    normalize_factors = (weights_to_combine != 0.0).double().sum(dim=1)
                    normalize_factors[normalize_factors == 0.] = torch.ones_like(normalize_factors[normalize_factors == 0.])
                    weights_combined = weights_to_combine.sum(dim=1) / normalize_factors
                    for sitc_idx, sitc in enumerate(synapse_indices_to_combine):
                        neurons_column_reduced[nln+'.weight'][:,column_index] = weights_combined
                    column_index += 1

        self.branch_in.load_state_dict({
            'weight':   neurons_column_reduced['branch_in.weight'],
            'bias':     neurons_column_reduced['branch_in.bias'],
        })
        self.layer_input.load_state_dict({
            'weight':   neurons_column_reduced['layer_input.weight'],
            'bias':     neurons_column_reduced['layer_input.bias'],
        })
        self.layer_hidden1.load_state_dict({
            'weight':   neurons_column_reduced['layer_hidden1.weight'],
            'bias':     neurons_column_reduced['layer_hidden1.bias'],
        })
        self.layer_hidden2.load_state_dict({
            'weight':   neurons_column_reduced['layer_hidden2.weight'],
            'bias':     neurons_column_reduced['layer_hidden2.bias'],
        })            
        self.ensemble_out.load_state_dict({
            'weight':   neurons_column_reduced['ensemble_out.weight'],
            'bias':     neurons_column_reduced['ensemble_out.bias'],
        })                           

    def forward(self, x):
        x = x.view(-1, x.shape[-3]*x.shape[-2]*x.shape[-1])
        x = self.branch_in(x)
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
        x = self.ensemble_out(x)
        return x


class MLP_trimmed_neuron(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, models, cluster_labels):
        super().__init__()
        neurons_fedavg = {
            'layer_input.weight':       torch.zeros(dim_hidden, dim_in),
            'layer_input.bias':         torch.zeros(dim_hidden),
            'layer_hidden1.weight':     torch.zeros(dim_hidden, dim_hidden),
            'layer_hidden1.bias':       torch.zeros(dim_hidden),
            'layer_hidden2.weight':     torch.zeros(dim_out, dim_hidden),
            'layer_hidden2.bias':       torch.zeros(dim_out),
        }
        self.alpha = {
            'layer_input.weight':       torch.zeros(dim_hidden, dim_in),
            'layer_input.bias':         torch.zeros(dim_hidden),
            'layer_hidden1.weight':     torch.zeros(dim_hidden, dim_hidden),
            'layer_hidden1.bias':       torch.zeros(dim_hidden),
            'layer_hidden2.weight':     torch.zeros(dim_out, dim_hidden),
            'layer_hidden2.bias':       torch.zeros(dim_out),   
        }
        for ln, cl in cluster_labels.items():
            for n_idx in range(cl.shape[0]):
                highest_cluster_idx = cluster_labels[ln][n_idx,:].max().astype(np.int)
                cluster_sizes = np.zeros(highest_cluster_idx+1)
                for hci in range(highest_cluster_idx+1):
                    cluster_sizes[hci] = (cluster_labels[ln][n_idx,:] == hci).sum()
                largest_cluster = np.argmax(cluster_sizes)
                largest_cluster_idx = np.argwhere(cluster_labels[ln][n_idx,:] == largest_cluster).flatten()
                self.alpha[ln+'.weight'][n_idx, :] = (float(len(largest_cluster_idx) - 1) / float(len(models) - 1)) * torch.ones_like(self.alpha[ln+'.weight'][n_idx, :])
                neuron_cluster = torch.zeros(len(largest_cluster_idx), neurons_fedavg[ln+'.weight'].shape[1])
                neuron_bias = 0.
                for lcii, lci in enumerate(largest_cluster_idx):
                    neuron_cluster[lcii] = models[lci].state_dict()[ln+'.weight'][n_idx,:]
                    neuron_bias += models[lci].state_dict()[ln+'.bias'][n_idx]
                neurons_fedavg[ln+'.weight'][n_idx] = neuron_cluster.mean(dim=0)
                neurons_fedavg[ln+'.bias'][n_idx] = neuron_bias / len(largest_cluster_idx)
            self.alpha[ln+'.bias'] = self.alpha[ln+'.weight'][:, 0]

        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_hidden2 = nn.Linear(dim_hidden, dim_out)
        self.relu = nn.ReLU()

        self.layer_input.load_state_dict({
            'weight':   neurons_fedavg['layer_input.weight'],
            'bias':   neurons_fedavg['layer_input.bias'],
        })
        self.layer_hidden1.load_state_dict({
            'weight':   neurons_fedavg['layer_hidden1.weight'],
            'bias':     neurons_fedavg['layer_hidden1.bias'],
        })
        self.layer_hidden2.load_state_dict({
            'weight':   neurons_fedavg['layer_hidden2.weight'],
            'bias':     neurons_fedavg['layer_hidden2.bias'],
        })                                     

    def forward(self, x):
        x = x.view(-1, x.shape[-3]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.layer_hidden1(x)
        x = self.relu(x)
        x = self.layer_hidden2(x)
        return x



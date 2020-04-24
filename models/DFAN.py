#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from __future__ import print_function

import torch
import copy
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable

import numpy as np
from sklearn.cluster import DBSCAN

from models.Fed import FedAvg

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

    for i in range(args.epoch_itrs):
        for k in range(1):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            optimizer_G.zero_grad()
            fake = generator(z)
            fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
            s_logit = student(fake)
            # t_sm = ensemble(teacher, fake, detach=False, mode=args.ensemble_mode)
            t_logit = torch.zeros_like(teacher[0](fake))
            for k in range(10):
                t_logit += teacher[k](fake)
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
            t_logit = torch.zeros_like(teacher[0](fake).detach())
            for k in range(10):
                t_logit += teacher[k](fake).detach()
            oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
            loss_S = torch.log(1. / (2. - oneMinus_P_S))
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
            t_sm_out += sm(t_l) * t_s.reshape(-1,1).repeat(1,10)
    elif mode == 0:
        t_sm_out = torch.zeros_like(t_logit[0])
        for t_l in t_logit:
            t_sm_out += sm(t_l) / 10.

    return t_sm_out



def DFAN_multigen(args, teacher, student, generator, optimizer, epoch):

    loss_G = []
    for ii in range(10):
        teacher[ii].eval()
        generator[ii].train()
        loss_G.append(torch.tensor(0.0))
    student.train()
    loss_S = torch.tensor(0.0)
    optimizer_S, optimizer_G = optimizer
    sm = torch.nn.Softmax()

    for i in range(args.epoch_itrs):
        for j in range(5):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            optimizer_S.zero_grad()
            oneMinus_P_S = []
            for ii in range(10):
                fake = generator[ii](z).detach()
                fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
                s_logit = student(fake)
                t_logit = teacher[ii](fake).detach()
                oneMinus_P_S.append(torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit))))
            loss_S = torch.log(1. / (2. - oneMinus_P_S[0]))
            for ii in range(1,10):
                loss_S += torch.log(1. / (2. - oneMinus_P_S[ii]))
            loss_S.backward()
            optimizer_S.step()
        for k in range(1):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            for ii in range(10):
                optimizer_G[ii].zero_grad()
                fake = generator[ii](z)
                fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
                s_logit = student(fake)
                t_logit = teacher[ii](fake)
                oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
                max_Gout = torch.max(torch.abs(fake))
                if max_Gout > 8.0:
                    loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0) + torch.pow(max_Gout - 8.0, 2.0)
                    print(max_Gout)
                else:
                    loss_G = torch.log(2.-oneMinus_P_S) + torch.pow(fake.var() - 1.0, 2.0)
                loss_G.backward()                   
                optimizer_G[ii].step()

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))        #vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            #vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())


def DFAN_single(args, teacher, student, generator, optimizer, epoch):

    teacher.eval()
    generator.train()
    student.train()
    loss_G = torch.tensor(0.0)
    loss_S = torch.tensor(0.0)
    optimizer_S, optimizer_G = optimizer
    sm = torch.nn.Softmax()

    for i in range(args.epoch_itrs):
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
            loss_S = torch.log(1. / (2. - oneMinus_P_S))
            loss_S.backward()
            optimizer_S.step()

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))        #vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            #vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())

def DFAN_regavg_hack(args, teacher, student, generator, optimizer, epoch):

    for local in teacher:
        local.eval()
    generator.train()
    student.train()
    loss_G = torch.tensor(0.0)
    loss_S = torch.tensor(0.0)
    optimizer_S, optimizer_G = optimizer
    sm = torch.nn.Softmax()

    w_locals = []
    for local in teacher:
        w_locals.append(copy.deepcopy(local.state_dict()))
    w_fedavg = FedAvg(w_locals)

    for i in range(args.epoch_itrs):
        for k in range(1):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            optimizer_G.zero_grad()
            fake = generator(z)
            fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
            s_logit = student(fake)
            # t_sm = ensemble(teacher, fake, detach=False, mode=args.ensemble_mode)
            t_logit = torch.zeros_like(teacher[0](fake))
            for k in range(10):
                t_logit += teacher[k](fake)
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
            t_logit = torch.zeros_like(teacher[0](fake).detach())
            for k in range(10):
                t_logit += teacher[k](fake).detach()
            oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
            loss_S1 = torch.log(1. / (2. - oneMinus_P_S))

            diff_L2 = torch.FloatTensor([0.]).to(args.device)
            alpha = float(max(min(epoch-10, 40), 0)) / 40.
            for student_w, fedavg_w in zip(student.parameters(), w_fedavg.values()):
                diff_L2 += ((fedavg_w - student_w)*fedavg_w.abs()).norm(2)
            loss_S2 = diff_L2 * alpha

            loss_S = loss_S1 + loss_S2
            loss_S.backward()
            optimizer_S.step()

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))        #vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            #vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())


def DFAN_regavg(args, teacher, student, generator, optimizer, epoch):

    for local in teacher:
        local.eval()
    generator.train()
    student.train()
    loss_G = torch.tensor(0.0)
    loss_S = torch.tensor(0.0)
    optimizer_S, optimizer_G = optimizer
    sm = torch.nn.Softmax()

    w_locals = []
    for local in teacher:
        w_locals.append(copy.deepcopy(local.state_dict()))
    w_fedavg = FedAvg(w_locals)

    for i in range(args.epoch_itrs):
        for k in range(1):
            z = torch.randn((args.batch_size, args.nz, 1, 1)).to(args.device)
            optimizer_G.zero_grad()
            fake = generator(z)
            fake = fake.view(-1, args.img_size[0], args.img_size[1], args.img_size[2])
            s_logit = student(fake)
            # t_sm = ensemble(teacher, fake, detach=False, mode=args.ensemble_mode)
            t_logit = torch.zeros_like(teacher[0](fake))
            for k in range(10):
                t_logit += teacher[k](fake) 
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
            t_logit = torch.zeros_like(teacher[0](fake).detach())
            for k in range(10):
                t_logit += teacher[k](fake).detach() 
            oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
            loss_S1 = torch.log(1. / (2. - oneMinus_P_S))

            diff_L2 = torch.FloatTensor([0.]).to(args.device)
            alpha = float(max(min(epoch-10, 40), 0)) / 40.
            for student_w, fedavg_w in zip(student.parameters(), w_fedavg.values()):
                diff_L2 += ((fedavg_w - student_w)*fedavg_w.abs()).norm(2)
            loss_S2 = diff_L2 * alpha

            loss_S = loss_S1 + loss_S2
            loss_S.backward()
            optimizer_S.step()

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))        #vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            #vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())



def model_fusion_MLP(models):
    eps = 0.02
    neurons = {
                'layer_input': torch.zeros(200, 10, 1024+1),
                'layer_hidden1': torch.zeros(200, 10, 200+1),
                'layer_hidden2': torch.zeros(10, 10, 200+1),
              }
    for mi, mnn in enumerate(models):
        for ln in neurons.keys():
            neurons[ln][:, mi, :] = torch.cat([mnn.state_dict()[ln+'.weight'], mnn.state_dict()[ln+'.bias'].unsqueeze(dim=1)], dim=1)
    cluster_labels = {
                        'layer_input': np.zeros((200, 10)),
                        'layer_hidden1': np.zeros((200, 10)),
                        'layer_hidden2': np.zeros((10, 10)),
                      }
    for ln, nl in neurons.items():
        for n_idx in range(nl.shape[0]):
            db = DBSCAN(eps=eps, min_samples=1).fit(nl[n_idx,:,:])
            cluster_labels[ln][n_idx,:] = db.labels_

    num_models = len(models)

    return MLP_agg(1024, 200, 10, num_models, neurons, cluster_labels)


class MLP_agg(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_models, neurons, cluster_labels):
        super().__init__()
        num_clusters, num_neurons = {},{}
        for ln, cl in cluster_labels.items():
            num_clusters[ln] = cl.max(axis=1) + 1
            num_neurons[ln] = num_clusters[ln].sum().astype(np.int)
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
        for mi in range(num_models):
            neurons_ensemble['branch_in.weight'][mi*dim_in:(mi+1)*dim_in, :] = torch.eye(dim_in)
            neurons_ensemble['layer_input.weight'][mi*dim_hidden:(mi+1)*dim_hidden, mi*dim_in:(mi+1)*dim_in] = neurons['layer_input'][:, mi, 0:dim_in]
            neurons_ensemble['layer_input.bias'][mi*dim_hidden:(mi+1)*dim_hidden] = neurons['layer_input'][:, mi, dim_in:(dim_in+1)].flatten()
            neurons_ensemble['layer_hidden1.weight'][mi*dim_hidden:(mi+1)*dim_hidden, mi*dim_hidden:(mi+1)*dim_hidden] = neurons['layer_hidden1'][:, mi, 0:dim_hidden]
            neurons_ensemble['layer_hidden1.bias'][mi*dim_hidden:(mi+1)*dim_hidden] = neurons['layer_hidden1'][:, mi, dim_hidden:(dim_hidden+1)].flatten()
            neurons_ensemble['layer_hidden2.weight'][mi*dim_out:(mi+1)*dim_out, mi*dim_hidden:(mi+1)*dim_hidden] = neurons['layer_hidden2'][:, mi, 0:dim_hidden]
            neurons_ensemble['layer_hidden2.bias'][mi*dim_out:(mi+1)*dim_out] = neurons['layer_hidden2'][:, mi, dim_hidden:(dim_hidden+1)].flatten()
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
                        neurons_row_reduced[ln+'.weight'][row_index][cmap_ci[nita_idx]*num_synapses_original[ln]:(cmap_ci[nita_idx]+1)*num_synapses_original[ln]] = weights_averaged
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
                    weights_combined = weights_to_combine.sum(dim=1) / (weights_to_combine != 0.0).double().sum(dim=1)
                    for sitc_idx, sitc in enumerate(synapse_indices_to_combine):
                        neurons_column_reduced[nln+'.weight'][:,column_index] = weights_combined
                    column_index += 1

        self.branch_in.load_state_dict({'weight':   neurons_column_reduced['branch_in.weight'],
                                        'bias':     neurons_column_reduced['branch_in.bias'],
        })
        self.layer_input.load_state_dict({'weight': neurons_column_reduced['layer_input.weight'],
                                          'bias':   neurons_column_reduced['layer_input.bias'],
        })
        self.layer_hidden1.load_state_dict({'weight':   neurons_column_reduced['layer_hidden1.weight'],
                                            'bias':     neurons_column_reduced['layer_hidden1.bias'],
        })
        self.layer_hidden2.load_state_dict({'weight':   neurons_column_reduced['layer_hidden2.weight'],
                                            'bias':     neurons_column_reduced['layer_hidden2.bias'],
        })            
        self.ensemble_out.load_state_dict({'weight':    neurons_column_reduced['ensemble_out.weight'],
                                           'bias':      neurons_column_reduced['ensemble_out.bias'],
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


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

    # mlpa, mlpt = model_fusion_MLP(teacher, 1024, 200, 10)
    # mlpa, mlpt = mlpa.to(args.device), mlpt.to(args.device)
    # w_mlpt = mlpt.state_dict()
    # alpha = mlpa.minavg_cluster_ratio * 3.0

    delta = torch.zeros(len(teacher),len(teacher))
    for ii, t1 in enumerate(teacher):
        t1_sd = t1.state_dict()
        for jj, t2 in enumerate(teacher):
            t2_sd = t2.state_dict()
            for ln in t1_sd.keys():
                delta[ii,jj] += (t1_sd[ln] - t2_sd[ln]).pow(2.0).sum()
    alpha = norm.cdf(-delta.mean(), scale=0.5) * 4.0

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
            # t_logit = mlpa(fake)
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
            # t_logit = mlpa(fake).detach()
            oneMinus_P_S = torch.tanh(F.kl_div(F.log_softmax(s_logit, dim=1), sm(t_logit)))
            loss_S1 = torch.log(1. / (2. - oneMinus_P_S))

            diff_L2 = torch.FloatTensor([0.]).to(args.device)
            # alpha = float(max(min(epoch-10, 40), 0)) / 40.
            for student_w, fedavg_w in zip(student.parameters(), w_fedavg.values()):
                diff_L2 += ((fedavg_w - student_w)*fedavg_w.abs()).norm(2)
            loss_S2 = diff_L2 * alpha

            # diff_L2 = torch.FloatTensor([0.]).to(args.device)
            # for student_w, mlpt_w in zip(student.parameters(), w_mlpt.values()):
            #     diff_L2 += ((mlpt_w - student_w)*mlpt_w.abs()).norm(2)
            # loss_S2 = diff_L2 * alpha

            loss_S = loss_S1 + loss_S2
            loss_S.backward()
            optimizer_S.step()

        if args.verbose and i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tG_Loss: {:.6f} S_loss: {:.6f}'.format(
                epoch, i, args.epoch_itrs, 100 * float(i) / float(args.epoch_itrs), loss_G.item(), loss_S.item()))        #vp.add_scalar('Loss_S', (epoch-1)*args.epoch_itrs+i, loss_S.item())
            #vp.add_scalar('Loss_G', (epoch-1)*args.epoch_itrs+i, loss_G.item())


def model_fusion_MLP(models, dim_in, dim_hidden, dim_out):
    eps_agg, eps_trimmed = 0.0001, 0.1
    num_models = len(models)

    neurons = {
        'layer_input': torch.zeros(dim_hidden, num_models, dim_in+1),
        'layer_hidden1': torch.zeros(dim_hidden, num_models, dim_hidden+1),
        'layer_hidden2': torch.zeros(dim_out, num_models, dim_hidden+1),
    }
    for mi, mnn in enumerate(models):
        for ln in neurons.keys():
            neurons[ln][:, mi, :] = torch.cat([mnn.state_dict()[ln+'.weight'], mnn.state_dict()[ln+'.bias'].unsqueeze(dim=1)], dim=1)
    cluster_labels_agg = {
        'layer_input': np.zeros((dim_hidden, num_models)),
        'layer_hidden1': np.zeros((dim_hidden, num_models)),
        'layer_hidden2': np.zeros((dim_out, num_models)),
    }
    cluster_labels_trimmed = copy.deepcopy(cluster_labels_agg)
    for ln, nl in neurons.items():
        for n_idx in range(nl.shape[0]):
            db_agg = DBSCAN(eps=eps_agg, min_samples=1).fit(nl[n_idx,:,:])
            cluster_labels_agg[ln][n_idx,:] = db_agg.labels_
            db_trimmed = DBSCAN(eps=eps_trimmed, min_samples=1).fit(nl[n_idx,:,:])            
            cluster_labels_trimmed[ln][n_idx,:] = db_trimmed.labels_

    delta = []
    for ii, m in enumerate(models):
        delt_layers = {}
        t_sd = m.state_dict()
        for ln in t_sd.keys():
            delt_layers[ln] = t_sd[ln]
        delta.append(delt_layers)

    return MLP_agg(dim_in, dim_hidden, dim_out, models, cluster_labels_agg), MLP_trimmed(dim_in, dim_hidden, dim_out, models, cluster_labels_trimmed)


class MLP_agg(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, models, cluster_labels):
        super().__init__()
        num_models = len(models)
        num_clusters, num_neurons = {},{}
        self.minavg_cluster_ratio = 1.0
        for ln, cl in cluster_labels.items():
            num_clusters[ln] = cl.max(axis=1) + 1
            num_neurons[ln] = num_clusters[ln].sum().astype(np.int)
            self.minavg_cluster_ratio = min((1.0 - (float(num_neurons[ln]) / cluster_labels[ln].shape[0] / num_models)), self.minavg_cluster_ratio)
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


class MLP_trimmed(nn.Module):
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
        self.minavg_cluster_ratio = 1.0
        for ln, cl in cluster_labels.items():
            avg_largest_cluster_idx = 0.0
            for n_idx in range(cl.shape[0]):
                highest_cluster_idx = cluster_labels[ln][n_idx,:].max().astype(np.int)
                cluster_sizes = np.zeros(highest_cluster_idx+1)
                for hci in range(highest_cluster_idx+1):
                    cluster_sizes[hci] = (cluster_labels[ln][n_idx,:] == hci).sum()
                largest_cluster = np.argmax(cluster_sizes)
                largest_cluster_idx = np.argwhere(cluster_labels[ln][n_idx,:] == largest_cluster).flatten()
                avg_largest_cluster_idx += float(len(largest_cluster_idx)) / float(len(models)) / float(cl.shape[0])
                neuron_cluster = torch.zeros(len(largest_cluster_idx), neurons_fedavg[ln+'.weight'].shape[1])
                neuron_bias = 0.
                for lcii, lci in enumerate(largest_cluster_idx):
                    neuron_cluster[lcii] = models[lci].state_dict()[ln+'.weight'][n_idx,:]
                    neuron_bias += models[lci].state_dict()[ln+'.bias'][n_idx]
                neurons_fedavg[ln+'.weight'][n_idx] = neuron_cluster.mean(dim=0)
                neurons_fedavg[ln+'.bias'][n_idx] = neuron_bias / len(largest_cluster_idx)
            self.minavg_cluster_ratio = min(self.minavg_cluster_ratio, avg_largest_cluster_idx)

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



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args, stop_at_batch=-1, shuffle=False):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=shuffle)
    if stop_at_batch == -1:
        dataset_len = len(data_loader) * args.bs
    else:
        dataset_len = stop_at_batch * args.bs
    for idx, (data, target) in enumerate(data_loader):
        if idx == stop_at_batch:
            break
        if args.device != torch.device('cpu'):
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= dataset_len
    accuracy = 100.00 * correct.float() / dataset_len
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, dataset_len, accuracy))
    return accuracy, test_loss

def test_img_ensem(net_locals, datatest, args, stop_at_batch=-1, shuffle=False):
    for net in net_locals:
        net.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs, shuffle=shuffle)
    if stop_at_batch == -1:
        dataset_len = len(data_loader) * args.bs
    else:
        dataset_len = stop_at_batch * args.bs
    for idx, (data, target) in enumerate(data_loader):
        if idx == stop_at_batch:
            break
        if args.device != torch.device('cpu'):
            data, target = data.cuda(), target.cuda()
        log_probs_list = []
        for net in net_locals:
            log_probs_list.append(net(data))
        log_probs_tensor = torch.stack(log_probs_list)
        log_probs = log_probs_tensor.mean(dim=0)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= dataset_len
    accuracy = 100.00 * correct.float() / dataset_len
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, dataset_len, accuracy))
    return accuracy, test_loss

# def test_img(net_g, datatest, args):
#     net_g.eval()
#     # testing
#     test_loss = 0.
#     correct = 0
#     data_loader = DataLoader(datatest, batch_size=args.bs)
#     loss_func = torch.nn.CrossEntropyLoss()
#     l = len(data_loader)
#     for idx, (data, target) in enumerate(data_loader):
#         if args.gpu != -1:
#             data, target = data.cuda(), target.cuda()
#         log_probs = net_g(data)
#         # sum up batch loss
#         test_loss += loss_func(log_probs, target).item()
#         # get the index of the max log-probability
#         y_pred = torch.argmax(log_probs, dim=1, keepdim=False) 
#         correct += sum(y_pred == target).int()

#     test_loss /= len(data_loader.dataset)
#     accuracy = 100.00 * correct.float() / len(data_loader.dataset)
#     if args.verbose:
#         print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
#             test_loss, correct, len(data_loader.dataset), accuracy))
#     return accuracy, test_loss

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=256, help="test batch size")
    parser.add_argument('--lr_device', type=float, default=0.1, help="Device local learning rate")
    parser.add_argument('--momentum', type=float, default=0.0, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--async_s2d', type=int, default=0, help='async server-to-device update across all devices or not (default 0: synchronous)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--weight_init', type=str, default='rand', help='weight initialization type. Recommended : xavier')
    parser.add_argument('--bias_init', type=str, default='rand', help='bias initialization type.  Recommended: zeros')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--store_models', type=str, default="testrun", help='Directory for model storage under deepedge/data')
    parser.add_argument('--noniid_hard', action='store_true')

    parser.add_argument('--lbfgs_hist', type=int, default=10, metavar='LH', help='memory size of LBFGS (default: 100)')
    parser.add_argument('--lr_server_qn', type=float, default=0.5, help="global learning rate for Quasi Newton")
    parser.add_argument('--lr_server_gd', type=float, default=1.0, help="global learning rate for gradient descent")
    parser.add_argument('--vr_mode', type=int, default=0, help="Variance reduction mode.  0: None, 1: SAGA-Scaffold, 2: Modified SAGA-Scaffold to halve uplink-bandwidth.")
    parser.add_argument('--opt_mode', type=int, default=0, help="Server-side optimization mode.  0: Plain vanilla model averaging only (FedAvg, Scaffold, etc), no qn.  1. qn only, no model averaging.  2: qn + model averaging.  3: Adaptive first order.")
    parser.add_argument('--vr_scale', type=float, default=1.0, help="Used with vr_mode = 1.  For SAG, set at 1/n where n is number of users.  For SAGA, set at default 1.0.")
    parser.add_argument('--max_qndn', type=float, default=1.0, help="Max quasi-newton step norm.")
    parser.add_argument('--adaptive_mode', type=int, default=0, help="Used with opt_mode = 3 only:  1. FedAdaGrad, 2. FedYogi, 3. FedAdam")

    parser.add_argument('--fedprox', type=float, default=0.0, help="default = off, to use, set to 1.0")
    parser.add_argument('--device_reg_norm2', type=float, default=0.0, help="default = off, to use, set to 1.0")
    parser.add_argument('--noniid_dirich_alpha', type=float, default=10.0, help="Default 10.0 is IID.  For nonIID, set to 0.5 to start with.")
    parser.add_argument('--adaptive_tau', type=float, default=0.1, help="Used with opt_mode = 3 only.")
    parser.add_argument('--adaptive_b1', type=float, default=0.9, help="Used with opt_mode = 3 only.")
    parser.add_argument('--adaptive_b2', type=float, default=0.99, help="Used with opt_mode = 3 only.")


    parser.add_argument('--screendump_file', type=str, default='', help="path to screen dump")


    args = parser.parse_args()
    return args

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
    parser.add_argument('--lr_server', type=float, default=0.1, help="Server global learning rate")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

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
    parser.add_argument('--linregress_numinputs', type=int, default=50)

    parser.add_argument('--noniid_dirich_alpha', type=float, default=10.0, help="Default 10.0 is IID.  For nonIID, set to 0.5 to start with.")

    parser.add_argument('--adaptive_tau', type=float, default=1e-08, help="Used with opt modes 1 through 3")
    parser.add_argument('--adaptive_b1', type=float, default=0.9, help="Used with opt modes 0 through 3")
    parser.add_argument('--adaptive_b2', type=float, default=0.999, help="Used with opt modes 1 through 3")
    parser.add_argument('--adaptive_bc', action='store_true', help="Bias correction for adaptive optim from Pytorch implementation")

    parser.add_argument('--client_opt_mode', type=int, default=0, help="")
    parser.add_argument('--client_momentum_mode', type=int, default=0, help="")
    parser.add_argument('--client_mime_lite', action='store_true', help="")

    parser.add_argument('--kronecker_lambda', type=float, default=0.04)
    parser.add_argument('--kronecker_mu1', type=float, default=0.2)
    parser.add_argument('--kronecker_beta', type=float, default=0.9)
    parser.add_argument('--kronecker_bc_off', action='store_true', help="Turn off bias correction for kronecker metrics momentum")
    parser.add_argument('--momentum_beta', type=float, default=0.9)
    parser.add_argument('--momentum_bc_off', action='store_true', help="Turn off bias correction for global and client gradient momentum")
    parser.add_argument('--kronecker_stop_update', type=int, default=-1, help="Provide round index to stop KBFGS update.")
    parser.add_argument('--warmup_dataset', type=str, default='', help="Name of warmup dataset for initial H_mat")

    parser.add_argument('--lenet5_activation', type=str, default='relu', help="Options: relu or tanh")
    parser.add_argument('--datasets_normalization', type=str, default='custom', help="Options: custom or generic.")
    parser.add_argument('--newton_method', type=str, default='predetermined', help="Options: predetermined or linesearch")

    parser.add_argument('--screendump_file', type=str, default='', help="path to screen dump")


    args = parser.parse_args()
    return args







def args_parser_core(parser):
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=256, help="test batch size")
    parser.add_argument('--lr_device', type=float, default=0.1, help="Device local learning rate")
    parser.add_argument('--lr_server', type=float, default=1.0, help="Server global learning rate")
    parser.add_argument('--lambda_reg', type=float, default=0.0, help="Ridge regularization")

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
    parser.add_argument('--linregress_numinputs', type=int, default=50)

    parser.add_argument('--noniid_dirich_alpha', type=float, default=10.0, help="Default 10.0 is IID.  For nonIID, set to 0.5 to start with.")

    parser.add_argument('--lenet5_activation', type=str, default='relu', help="Options: relu or tanh")
    parser.add_argument('--datasets_normalization', type=str, default='custom', help="Options: custom or generic.")

    parser.add_argument('--screendump_file', type=str, default='', help="path to screen dump")


def args_parser_fedsigmaxi():
    parser = argparse.ArgumentParser()

    args_parser_core(parser)
    parser.add_argument('--num_local_steps', type=int, default=10)
    parser.add_argument('--grad_ref_alpha', type=float, default=0.9)
    parser.add_argument('--use_grad_for_ref', action='store_true')
    parser.add_argument('--dynamic_batch_size', action='store_true')
    parser.add_argument('--sigma_est_samples', type=int, default=5)
    parser.add_argument('--use_local_gradref_mom', action='store_true')
    parser.add_argument('--dynamic_lr', action='store_true')
    parser.add_argument('--hyper_lrlr', type=float, default=0.001, help='Hyper-learning rate for dynamic learning rate.')
    parser.add_argument('--use_grad_for_dlr', action='store_true')

    args = parser.parse_args()
    return args
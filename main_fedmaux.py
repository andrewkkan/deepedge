#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# Federated Minimum Auxiliary Gradient Algorithm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, mnist_sample_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.test import test_img, test_img_ensem
from models.Fed import FedAvgMomentum

from IPython import embed

if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not os.path.exists('data/models/%s'%(args.store_models)):
        os.makedirs('data/models/%s'%(args.store_models))

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            # dict_users = mnist_iid(dataset_train, args.num_users)
            dict_users = mnist_sample_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.noniid_hard)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')

    else:
        exit('Error: unrecognized dataset')
    args.img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        net_glob = MLP(dim_in=args.img_size[0]*args.img_size[1]*args.img_size[2], dim_hidden=200,
                       dim_out=args.num_classes,
                       weight_init=args.weight_init, bias_init=args.bias_init)
        # optimizer_glob = torch.optim.SGD(net_glob.parameters(), lr=args.lr_S,
        #                                  weight_decay=args.weight_decay, momentum=0.0)
        net_glob = net_glob.to(args.device)
        net_glob.train()
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    local_user = []
    for idx in range(args.num_users):
        local_user.append(LocalUpdate(args=args, net=copy.deepcopy(net_glob).to(args.device), dataset=dataset_train, idxs=dict_users[idx]))
    last_update = np.ones(args.num_users) * -1
    edge_update = -1

    history_glob = []
    v_sd = copy.deepcopy(net_glob.state_dict())
    vhat_sd = copy.deepcopy(net_glob.state_dict())
    subepoch_idx = 0

    for epoch_idx in range(args.epochs):
        loss_locals, acc_locals, acc_locals_on_local = [], [], []

        if args.rand_d2s == 0.0: # default
            m = min(max(int(args.frac * args.num_users), 1), args.num_users)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        else:
            if args.rand_d2s == []:
                frac_list = [args.frac]
            else:
                frac_list = args.rand_d2s
            rand_d2s = np.resize(frac_list, args.num_users)
            idxs_users = np.where(np.random.random(args.num_users) < rand_d2s)[0]
            m = len(idxs_users)
            if m == 0:
                m = 1
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        if args.sync2async > 0:
            if epoch_idx > args.sync2async:
                args.async_s2d = 1
            else:
                args.async_s2d = 0

        for iu_idx, user_idx in enumerate(idxs_users):
            use_aux = True
            cnet_locals, lnet_locals = [], []
            if args.async_s2d == 1:  # async mode 1 updates after FedAvg (see lines below FedAvg)
                w, loss, acc_ll = local_user[user_idx].train()
                cnet_locals.append(copy.deepcopy(local_user[user_idx].net.state_dict()))
                lnet_locals.append(copy.deepcopy(local_user[user_idx].last_net.state_dict()))
            elif args.async_s2d == 2:  # async mode 2 updates before FedAvg
                w, loss, acc_ll = local_user[user_idx].train()
                cnet_locals.append(copy.deepcopy(local_user[user_idx].net.state_dict()))
                lnet_locals.append(copy.deepcopy(local_user[user_idx].last_net.state_dict()))
                local_user[user_idx].weight_update(net=copy.deepcopy(net_glob).to(args.device))
            elif args.async_s2d == 0:  # synchronous mode, updates before training
                local_user[user_idx].weight_update(net=copy.deepcopy(net_glob).to(args.device))
                last_update[user_idx] = epoch_idx
                w, loss, acc_ll = local_user[user_idx].train()
                cnet_locals.append(copy.deepcopy(local_user[user_idx].net.state_dict()))
                lnet_locals.append(copy.deepcopy(local_user[user_idx].last_net.state_dict()))
            acc_l, _ = test_img(local_user[user_idx].net, dataset_train, args, stop_at_batch=16, shuffle=True)
            loss_locals.append(loss)
            acc_locals.append(acc_l)
            acc_locals_on_local.append(acc_ll)

            # Check if this user meets timing requiement
            cnet_sd, lnet_sd = cnet_locals[0], lnet_locals[0]
            if last_update[user_idx] > edge_update:
                edge_update = last_update[user_idx]
            if last_update[user_idx] >= (edge_update - args.fedmaux_tau):
                vhat_sd = copy.deepcopy(v_sd)            
                w_fedavg_hat = FedAvgMomentum([cnet_sd], [lnet_sd], copy.deepcopy(net_glob.state_dict()), vhat_sd, args.server_momentum, subepoch_idx, scale=[args.server_lr])
                history_glob.append({
                    'glob_sd':      copy.deepcopy(w_fedavg_hat),
                    'timestamp':    epoch_idx,
                })
                if not args.fedmaux_greedy:
                    use_aux = False
                else:
                    use_aux = True
            else:
                w_fedavg_hat = None
                use_aux = True

            if use_aux == True:
                min_dist = 0.
                for hli, hg in enumerate(history_glob):
                    dist = 0.
                    for k in hg['glob_sd'].keys():
                        dist += (cnet_sd[k] - hg['glob_sd'][k]).pow(2.).sum()
                    if dist < min_dist or hli == 0:
                        min_dist = dist
                        ns_ref = hg['glob_sd']
                w_fedmaux = FedAvgMomentum([cnet_sd], [ns_ref], copy.deepcopy(net_glob.state_dict()), v_sd, args.server_momentum, subepoch_idx, scale=[args.server_lr])
                if w_fedavg_hat:
                    history_glob[-1] = {
                        'glob_sd':      copy.deepcopy(w_fedmaux),
                        'timestamp':    epoch_idx,
                    }
                else:
                    history_glob.append({
                        'glob_sd':      copy.deepcopy(w_fedmaux),
                        'timestamp':    epoch_idx,
                    })
            else:
                v_sd = vhat_sd

            net_glob.load_state_dict(history_glob[-1]['glob_sd'])

            if args.async_s2d == 1:  # async mode 1 updates after FedAvg
                local_user[user_idx].weight_update(net=copy.deepcopy(net_glob).to(args.device))
                last_update[user_idx] = epoch_idx

            if args.async_s2d == 2:
                # This goes last in the inner-loop after all last_update[user_idx] checks have been made
                last_update[user_idx] = epoch_idx

            subepoch_idx += 1

        # Calculate accuracy for each round
        acc_glob, _ = test_img(net_glob, dataset_test, args, shuffle=True)
        acc_loc = sum(acc_locals) / len(acc_locals)
        acc_lloc = 100. * sum(acc_locals_on_local) / len(acc_locals_on_local)

        # print status
        loss_avg = sum(loss_locals) / len(loss_locals)
        print(
                'Round {:3d}, Devices participated {:2d}, Average loss {:.3f}, Central accuracy on global test data {:.3f}, Local accuracy on global train data {:.3f}, Local accuracy on local train data {:.3f}\n\n'.\
                format(epoch_idx, m, loss_avg, acc_glob, acc_loc, acc_lloc)
        )
        loss_train.append(loss_avg)

        # Memory Management
        # Figure out worst straggler last_update
        worst_last_update = last_update[0]
        for idx in range(args.num_users):
            if last_update[idx] < worst_last_update:
                worst_last_update = last_update[idx]
        # Delete history_glob with timestamp before worst straggler last update
        while history_glob[0]['timestamp'] < worst_last_update:
            del history_glob[0]

        # for idx, user_idx in enumerate(idxs_users):
        #     torch.save(cnet_locals[idx].state_dict(),"data/models/%s/netlocal%s-epoch%s-model%s-dataset%s.pt"%(args.store_models,str(idx),str(epoch_idx),args.model,args.dataset))
        #     print(idx, user_idx,
        #           local_user[user_idx].labels)
        # torch.save(net_glob.state_dict(),"data/models/%s/netglob-epoch%s-model%s-dataset%s.pt"%(args.store_models,str(epoch_idx),args.model,args.dataset))
        # torch.save(generator.state_dict(),"data/models/%s/netgen-epoch%s-dataset%s.pt"%(args.store_models,str(epoch_idx),args.dataset))
        
    # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./log/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args, shuffle=True)
    #print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy on test data: {:.2f}, Testing loss: {:.2f}".format(acc_test, loss_test))

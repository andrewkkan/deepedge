#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms

from IPython import embed

def mnist_iid(dataset, num_users):
    """
    Partitions I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users, noniid_hard=False):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if noniid_hard:
        num_shards, num_imgs, shard_select = 100, 600, 1
    else:
        num_shards, num_imgs, shard_select = 200, 300, 2
    # num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shard_select, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def generic_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # dict_users[i] = set(all_idxs[0:num_items])
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def generic_noniid(dataset, num_users, alpha):
    """
    Basically very similar to what the following paper did for CIFAR100 course labels 
    "Adaptive Federated Optimization" by Reddi el al (Google)
    https://arxiv.org/pdf/2003.00295.pdf (Appendex F)
    """
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}    
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((np.arange(len(dataset)), labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    num_labels = len(dataset.classes)
    num_items = int(len(dataset)/num_users)
    for i in range(num_users):
        sel_dist = np.random.dirichlet(tuple(np.repeat(alpha, num_labels)))
        sel_cum = np.vstack((np.cumsum(sel_dist), np.arange(num_labels)))
        for item_idx in range(num_items):
            locate_label = []
            while (not locate_label) and idxs_labels.any():
                locate_sel = np.min(np.where(sel_cum[0] >= np.random.rand()))
                sel_label = sel_cum[1][locate_sel]
                locate_label = list(np.where(idxs_labels[1] == sel_label)[0])
                if not locate_label:
                    sel_dist = sel_dist[(sel_cum[1] != sel_label)] 
                    renorm = sel_dist.sum()
                    sel_cum = sel_cum[:, (sel_cum[1] != sel_label)]
                    sel_dist = sel_dist / renorm
                    sel_cum[0] = sel_cum[0] / renorm
            sel_sample = np.random.choice(locate_label, 1)
            dict_users[i] = np.append(dict_users[i], idxs_labels[0, sel_sample])
            idxs_labels = np.delete(idxs_labels, sel_sample, 1)
    return dict_users

def cifar100_noniid(dataset, num_users, alpha):
    """
    Basically very similar to what the following paper did for CIFAR100 course labels 
    "Adaptive Federated Optimization" by Reddi el al (Google)
    https://arxiv.org/pdf/2003.00295.pdf (Appendex F)
    """
    coarse_to_fine_label_mapping = {
        'aquatic mammals':                     ['beaver', 'dolphin', 'otter', 'seal', 'whale',],
        'fish':                                ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',],
        'flowers':                             ['orchid', 'poppy', 'rose', 'sunflower', 'tulip',],
        'food containers':                     ['bottle', 'bowl', 'can', 'cup', 'plate',],
        'fruit and vegetables':                ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',],
        'household electrical devices':        ['clock', 'keyboard', 'lamp', 'telephone', 'television',],
        'household furniture':                 ['bed', 'chair', 'couch', 'table', 'wardrobe',],
        'insects':                             ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',],
        'large carnivores':                    ['bear', 'leopard', 'lion', 'tiger', 'wolf',],
        'large man-made outdoor things':       ['bridge', 'castle', 'house', 'road', 'skyscraper',],
        'large natural outdoor scenes':        ['cloud', 'forest', 'mountain', 'plain', 'sea',],
        'large omnivores and herbivores':      ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',],
        'medium-sized mammals':                ['fox', 'porcupine', 'possum', 'raccoon', 'skunk',],
        'non-insect invertebrates':            ['crab', 'lobster', 'snail', 'spider', 'worm',],
        'people':                              ['baby', 'boy', 'girl', 'man', 'woman',],
        'reptiles':                            ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',],
        'small mammals':                       ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',],
        'trees':                               ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',],
        'vehicles 1':                          ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train',],
        'vehicles 2':                          ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor',],
    }

    coarse_fine_idxs_mapping = np.empty((2,0))
    coarse_idx = 0
    for coarse_label, fine_labels in coarse_to_fine_label_mapping.items():
        idx_mapping = []
        for fine_label in fine_labels:
            idx_mapping.append(dataset.class_to_idx[fine_label])
        coarse_fine_idxs_mapping = np.append(
            coarse_fine_idxs_mapping.transpose(), 
            np.vstack((np.repeat(coarse_idx, len(idx_mapping)), np.array(idx_mapping))).transpose(), 
            axis=0
        ).transpose()
        coarse_idx += 1
    coarse_fine_idxs_mapping = coarse_fine_idxs_mapping.astype(int)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}    
    fine_labels = np.array(dataset.targets)
    idx_mapper = lambda label, mapping: mapping[0][ np.argwhere( mapping[1] == label )[0][0] ]
    vector_mapper = np.vectorize(idx_mapper, excluded=['mapping'])
    labels = vector_mapper(label=fine_labels, mapping=coarse_fine_idxs_mapping)
    idxs_labels = np.vstack((np.arange(len(dataset)), labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]
    num_labels = len(coarse_to_fine_label_mapping.keys())
    num_items = int(len(dataset)/num_users)
    for i in range(num_users):
        sel_dist = np.random.dirichlet(tuple(np.repeat(alpha, num_labels)))
        sel_cum = np.vstack((np.cumsum(sel_dist), np.arange(num_labels)))
        for item_idx in range(num_items):
            locate_label = []
            while (not locate_label) and idxs_labels.any():
                locate_sel = np.min(np.where(sel_cum[0] >= np.random.rand()))
                sel_label = sel_cum[1][locate_sel]
                locate_label = list(np.where(idxs_labels[1] == sel_label)[0])
                if not locate_label:
                    sel_dist = sel_dist[(sel_cum[1] != sel_label)] 
                    renorm = sel_dist.sum()
                    sel_cum = sel_cum[:, (sel_cum[1] != sel_label)]
                    sel_dist = sel_dist / renorm
                    sel_cum[0] = sel_cum[0] / renorm
            sel_sample = np.random.choice(locate_label, 1)
            dict_users[i] = np.append(dict_users[i], idxs_labels[0, sel_sample]).astype(int)
            idxs_labels = np.delete(idxs_labels, sel_sample, 1)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)

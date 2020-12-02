import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import copy

from utils.sampling import mnist_iid, mnist_noniid, generic_iid, generic_noniid, cifar100_noniid
from utils.emnist_dataset import EMNISTDataset_by_write
from models.linRegress import DataLinRegress, lin_reg
from IPython import embed

def get_datasets(args):
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
            # dict_users = mnist_sample_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.noniid_hard)
        args.num_classes = 10
        args.num_channels = 1
        args.task = 'ObjRec'
    elif args.dataset == 'emnist':
        # trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.9635,), (0.1586,))])
        trans_emnist = transforms.Compose([transforms.ToTensor()])
        # dataset_train = EMNISTDataset_by_write(train=True, transform=trans_emnist)
        dataset_train = EMNISTDataset_by_write(train=True, transform=trans_emnist)
        dataset_test = EMNISTDataset_by_write(train=False, transform=trans_emnist)
        # sample users
        args.num_classes = 62
        args.num_channels = 1
        args.task = 'AutoEnc'
        args.model = 'autoenc'
        if args.num_users == 1:
            args.iid = True
            args.num_users = 1
            args.frac = 1.0
            dict_users = [list(range(len(dataset_train)))]
        else:
            args.iid = False
            args.num_users = 3500
            args.frac = 0.00286
            dict_users = dataset_train.dict_users
    elif args.dataset == 'reddit':
        args.frac = 0.0123
    elif args.dataset == 'cifar10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 10
        args.task = 'ObjRec'
    elif args.dataset == 'cifar100':
        trans_cifar100 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar100)
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar100)
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 100
        args.task = 'ObjRec'
    elif args.dataset == 'bcct200':
        if args.num_users > 8:
            args.num_users = 8
            print("Warning: limiting number of users to 8.")
        trans_bcct = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=[0.262428402,0.262428402,0.262428402], std=[0.30702864, 0.30702864, 0.30702864])])
        dataset_train = datasets.ImageFolder(root='./data/BCCT200_resized/', transform=trans_bcct)
        dataset_test = dataset_train
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 4
        args.task = 'ObjRec'
    elif args.dataset == 'linregress':
        args.linregress_numinputs = 50
        args.num_classes = 1
        dataset_train = DataLinRegress(linregress_numinputs, num_outputs=args.num_classes)
        dataset_test = dataset_train
        args.model = 'linregress'
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.task = 'LinReg'
    elif args.dataset == 'linsaddle':
        linregress_numinputs = 2
        args.num_classes = 2
        dataset_train = DataLinRegress(linregress_numinputs, noise_sigma=0.0, num_outputs=args.num_classes)
        dataset_test = dataset_train
        args.model = 'linregress'
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.task = 'LinSaddle'    
    else:
        exit('Error: unrecognized dataset')

    args.img_size = dataset_train[0][0].shape

    return dataset_train, dataset_test, dict_users, args
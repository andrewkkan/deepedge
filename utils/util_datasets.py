import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import copy

from utils.sampling import mnist_iid, mnist_noniid, generic_iid, generic_noniid, cifar100_noniid
from utils.emnist_dataset import EMNISTDataset_by_write
from models.linRegress import DataLinRegress, lin_reg
from IPython import embed

def get_warmup_datasets(args, image_dim):
    if image_dim == tuple((32, 32)):
        transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    if args.warmup_dataset == 'mnist':
        dataset_warmup = datasets.MNIST('./data/mnist/', train=True, download=True, transform=transform)
    elif args.warmup_dataset == 'emnist':
        dataset_warmup = datasets.EMNIST('./data/emnist/', train=True, download=True, transform=transform)
    elif args.warmup_dataset == 'fashionmnist':
        dataset_warmup = datasets.FashionMNIST('./data/fashionmnist/', train=True, download=True, transform=transform)
    elif args.warmup_dataset == 'cityscapes':
        dataset_warmup = datasets.Cityscapes('./data/cityscapes/', train=True, download=True, transform=transform)
    elif args.warmup_dataset == 'cifar10':
        dataset_warmup = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transform)
    elif args.warmup_dataset == 'cifar100':
        dataset_warmup = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=transform)
    elif args.warmup_dataset == 'bcct200':
        dataset_warmup = datasets.ImageFolder(root='./data/BCCT200_resized/', transform=transform)
    return dataset_warmup


def augment_num_channels(batchdata, num_channels):
    # batchdata.size()[0] is num batches, [1] is num channels, [2:3] are image HxL
    if batchdata.shape[1] == 1 and num_channels == 3:
        return torch.cat((batchdata, batchdata, batchdata), dim=1)
    elif batchdata.shape[1] == 3 and num_channels == 1:
        return batchdata[:, 0, :, :]
    else:
        return None

def get_datasets(args):
    trans_generic = [transforms.ToTensor()]
    trans_generic_2828 = [transforms.Resize((28, 28)), transforms.ToTensor()]
    trans_generic_3232 = [transforms.Resize((32, 32)), transforms.ToTensor()]

    if args.model == 'cnn': # CIFAR
        transform_list = trans_generic_3232
    elif args.model == 'lenet5':
        transform_list = trans_generic_3232
    # elif args.model == 'autoenc':
    #     transform = trans_generic_2828
    else:
        transform = trans_generic

    if args.dataset == 'mnist':
        if args.datasets_normalization == 'custom':
            transform_list.append(
                transforms.Normalize((0.1307,), (0.3081,))
            )
        else:
            transform_list.append(
                transforms.Normalize((0.5,), (0.25,))
            )            
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=transforms.Compose(transform_list))
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=transforms.Compose(transform_list))
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
            # dict_users = mnist_sample_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.noniid_hard)
        args.num_classes = 10
        args.num_channels = 1
    elif args.dataset == 'emnist':
        # trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.9635,), (0.1586,))])
        # dataset_train = EMNISTDataset_by_write(train=True, transform=trans_emnist)
        if args.datasets_normalization == 'custom':
            transform_list.append(
                transforms.Normalize((0.9637268,), (0.15913315,))
            )
        else:
            transform_list.append(
                transforms.Normalize((0.5,), (0.25,))
            )         
        dataset_train = EMNISTDataset_by_write(train=True, transform=transforms.Compose(transform_list))
        dataset_test = EMNISTDataset_by_write(train=False, transform=transforms.Compose(transform_list))
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
            # args.frac = 0.00286 # for 10 active users each round
            dict_users = dataset_train.dict_users
    elif args.dataset == 'reddit':
        args.frac = 0.0123
    elif args.dataset == 'cifar10':
        if args.datasets_normalization == 'custom':
            transform_list.append(
                # transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            )
        else:
            transform_list.append(
                transforms.Normalize((0.5,0.5,0.5), (0.25,0.25,0.25))
            )         
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=transforms.Compose(transform_list))
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=transforms.Compose(transform_list))
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        if args.datasets_normalization == 'custom':
            transform_list.append(
                transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047))
            )
        else:
            transform_list.append(
                transforms.Normalize((0.5,0.5,0.5), (0.25,0.25,0.25))
            )         
        dataset_train = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=transforms.Compose(transform_list))
        dataset_test = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=transforms.Compose(transform_list))
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar100_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 100
    elif args.dataset == 'bcct200':
        if args.num_users > 8:
            args.num_users = 8
            print("Warning: limiting number of users to 8.")
        if args.datasets_normalization == 'custom':
            transform_list.append(
                transforms.Normalize(mean=[0.262428402,0.262428402,0.262428402], std=[0.30702864, 0.30702864, 0.30702864])
            )
        else:
            transform_list.append(
                transforms.Normalize((0.5,0.5,0.5), (0.25,0.25,0.25))
            )         
        dataset_train = datasets.ImageFolder(root='./data/BCCT200_resized/', transform=transforms.Compose(transform_list))
        dataset_test = dataset_train
        if args.iid:
            dict_users = generic_iid(dataset_train, args.num_users)
        else:
            dict_users = generic_noniid(dataset_train, args.num_users, args.noniid_dirich_alpha)
        args.num_classes = 4
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
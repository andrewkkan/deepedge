import torch
import copy

from models.Nets import MLP, CNNMnist, CNNCifar, LeNet5, MNIST_AE
from models.linRegress import lin_reg


def get_model(args):
    if args.model == 'cnn' and args.dataset != 'mnist':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'lenet5' and args.dataset != 'mnist':
        net_glob = LeNet5(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        net_glob = MLP(dim_in=args.img_size[0]*args.img_size[1]*args.img_size[2], dim_hidden=200,
                       dim_out=args.num_classes,
                       weight_init=args.weight_init, bias_init=args.bias_init).to(args.device)
    elif args.model == 'linregress':    
        net_glob = lin_reg(args.linregress_numinputs, args.num_classes).to(args.device)
    elif args.model == 'autoenc':
        net_glob = MNIST_AE(dim_in = args.img_size[0]*args.img_size[1]*args.img_size[2])
    else:
        exit('Error: unrecognized model')

    return net_glob
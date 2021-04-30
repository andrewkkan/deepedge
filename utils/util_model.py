import torch
import copy


def get_model(args):
    from models.Nets import MLP, CNNMnist, CNNCifar, LeNet5, MNIST_AE
    from models.linRegress import lin_reg

    if args.model == 'cnn' and args.dataset != 'mnist':
        net_glob = CNNCifar(args=args).to(args.device)
        args.task = 'ObjRec'
    elif args.model == 'lenet5' and args.dataset != 'mnist':
        net_glob = LeNet5(args=args).to(args.device)
        args.task = 'ObjRec'
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        args.task = 'ObjRec'
    elif args.model == 'mlp':
        net_glob = MLP(dim_in=args.img_size[0]*args.img_size[1]*args.img_size[2], dim_hidden=200,
                       dim_out=args.num_classes,
                       weight_init=args.weight_init, bias_init=args.bias_init).to(args.device)
        args.task = 'ObjRec'
    elif args.model == 'linregress':    
        net_glob = lin_reg(args.linregress_numinputs, args.num_classes).to(args.device)
        assert(args.task == 'LinReg' or args.task == 'LinSaddle' )
    elif args.model == 'autoenc':
        net_glob = MNIST_AE(dim_in = args.img_size[0]*args.img_size[1]*args.img_size[2])
        args.task = 'AutoEnc'
    else:
        exit('Error: unrecognized model')

    return net_glob



def get_model_k(args):
    from models.Nets_K import MLP, CNNMnist, CNNCifar, LeNet5, MNIST_AE

    if args.model == 'cnn' and args.dataset != 'mnist':
        net_glob = CNNCifar(args=args).to(args.device)
        args.task = 'ObjRec'
    elif args.model == 'lenet5' and args.dataset != 'mnist':
        net_glob = LeNet5(args=args).to(args.device)
        args.task = 'ObjRec'
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
        args.task = 'ObjRec'
    elif args.model == 'mlp':
        net_glob = MLP(dim_in=args.img_size[0]*args.img_size[1]*args.img_size[2], dim_hidden=200,
                       dim_out=args.num_classes,
                       weight_init=args.weight_init, bias_init=args.bias_init).to(args.device)
        args.task = 'ObjRec'
    # elif args.model == 'linregress':    
    #     net_glob = lin_reg(args.linregress_numinputs, args.num_classes).to(args.device)
    elif args.model == 'autoenc':
        net_glob = MNIST_AE(dim_in = args.img_size[0]*args.img_size[1]*args.img_size[2])
        args.task = 'AutoEnc'
    else:
        exit('Error: unrecognized model')

    return net_glob


def gather_flat_params(model):
    flat_params = []
    for p in model.parameters():
        flat_params.append(p.data.view(-1))
    return torch.cat(flat_params, 0)

def gather_flat_params_with_grad(model):
    flat_params = []
    for p in model.parameters():
        flat_params.append(p.view(-1))
    return torch.cat(flat_params, 0)

def gather_flat_other_states(model):
    sd = model.state_dict()
    pd = dict(model.named_parameters())
    osl = []
    for sdk in sd.keys():
        if sdk not in pd.keys():
            osl.append(sd[sdk].data.view(-1).float())
    if osl:
        return torch.cat(osl, 0)
    else:
        return None

def gather_flat_states(model):
    flat_states = []
    for s in model.state_dict().values():
        flat_states.append(s.data.view(-1).float())
    return torch.cat(flat_states, 0)    

def add_states(model, flat_states):
    sd = model.state_dict()
    offset = 0
    for sdk in sd.keys():
        numel = sd[sdk].numel()
        if 'Long' in sd[sdk].type():
            sd[sdk] += flat_states[offset:offset + numel].view_as(sd[sdk]).long()
        else:
            sd[sdk] += flat_states[offset:offset + numel].view_as(sd[sdk])
        offset += numel

def gather_flat_grad(model):
    views = []
    for p in model.parameters():
        if p.grad is None:
            view = p.data.new(p.data.numel()).zero_()
        elif p.grad.data.is_sparse:
            view = p.grad.data.to_dense().view(-1)
        else:
            view = p.grad.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def net_params_halper(net, states):
    sdk = net.state_dict().keys()
    npk = dict(net.named_parameters()).keys()    
    wl, osl = [], []
    deltw, deltos = None, None
    offset = 0
    for k in sdk:
        numel = net.state_dict()[k].numel()
        if k in npk:
            wl.append(delts[offset:offset+numel])
        else:
            osl.append(delts[offset:offset+numel])
        offset += numel
    deltw = torch.cat(wl, 0)
    if osl:
        deltos = torch.cat(osl, 0)
    return deltw, deltos


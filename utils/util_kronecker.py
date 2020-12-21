import torch
import copy

from models.Nets_K import MLP, CNNMnist, CNNCifar, LeNet5, MNIST_AE
#from models.linRegress import lin_reg

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
    # elif args.model == 'linregress':    
    #     net_glob = lin_reg(args.linregress_numinputs, args.num_classes).to(args.device)
    elif args.model == 'autoenc':
        net_glob = MNIST_AE(dim_in = args.img_size[0]*args.img_size[1]*args.img_size[2])
    else:
        exit('Error: unrecognized model')

    return net_glob

def initialize_Hmat(net_sd):
    Hmat = []
    for layerkey, layerval in net_sd.items():
        if len(layerval.shape) == 2: # Look for fully connected matrices.  Add 1 for constant bias term .
            Hmat.append({
                'Hg': torch.eye(layerval.shape[0], device=layerval.device, requires_grad=False),
                'Ha': torch.eye(layerval.shape[1]+1, device=layerval.device, requires_grad=False),
                'sg': torch.zeros(layerval.shape[0], device=layerval.device, requires_grad=False),
                'yg': torch.zeros(layerval.shape[0], device=layerval.device, requires_grad=False),
                'A' : torch.zeros((layerval.shape[1]+1, layerval.shape[1]+1), device=layerval.device, requires_grad=False),
            })
        elif len(layerval.shape) == 4: # conv layers
            Hmat.append(None)
    return Hmat


def multiply_HgDeltHa(delt_w, H_mat, net_sd, device):
    # This function does matrix multiplication layer-by-layer: Hg x delt x Ha
    # It outputs a vectorized (flat) gradient 
    descent_list = []
    delt_idx = 0
    Hmat_idx = 0
    for layerkey, layerval in net_sd.items():
        if len(layerval.shape) == 2: # Look for fully connected matrices.  Add 1 for constant bias term .
            delt_layer_w = delt_w[delt_idx:delt_idx+(layerval.shape[0] * layerval.shape[1])].view_as(layerval)
            delt_idx += layerval.shape[0] * layerval.shape[1]
            delt_layer_b = delt_w[delt_idx:delt_idx+layerval.shape[0]].view(-1, 1)
            delt_idx += layerval.shape[0]
            delt_layer = torch.cat([delt_layer_w, delt_layer_b], dim=1) # m x (n+1) matrix
            descent = torch.mm(H_mat[Hmat_idx]['Hg'], delt_layer).data # Hg is m x m
            descent = torch.mm(descent, H_mat[Hmat_idx]['Ha']).data # Ha is (n+1) x (n+1)
            # The resulting descent from the above matrix multiplications has dim m x (n+1)
            descent_w = descent[:, 0:layerval.shape[1]].flatten()
            descent_b = descent[:, layerval.shape[1]].flatten()
            descent = torch.cat([descent_w, descent_b])
            descent_list.append(descent)
            Hmat_idx += 1
            del delt_layer_w
            del delt_layer_b
            del descent_w
            del descent_b
            del descent
        elif len(layerval.shape) == 4: # conv layers
            num_params = torch.tensor(layerval.shape).prod() + layerval.shape[0]
            descent = delt_w[delt_idx:delt_idx+num_params].view(-1)
            delt_idx += num_params
            Hmat_idx += 1
            descent_list.append(descent)
            del num_params
            del descent
    descent_vec = torch.cat(descent_list)
    del descent_list[:]
    return descent_vec

def get_s_sgrad(s):
    # Input s is a per-layer list
    # Each s_l is (bs, m_l) tensor vector as the pre-activation output,
    # where bs is batch size.
    s_list, sgrad_list = [], []
    for s_l in s:
        if s_l is None:
            s_list.append(None)
            sgrad_list.append(None)
            continue
        # Since s's are just outputs of the linear blocks from batch samples, 
        # and not affected by the scaling of the loss value, they have not been averaged with batchsize like s.grad.
        # Therefore, we need to take the mean of these values.
        s_list.append(s_l.mean(dim=0))
        # Needs to sum along the batchsize dimension (not the mean because the loss value has been averaged with batchsize).
        # Therefore, these resulting grad values represent the mean grad per sample
        sgrad_list.append(s_l.grad.sum(dim=0))
    return s_list, sgrad_list

def get_aaT_abar(a):
    # Input a is a per-layer list
    # Each a is (bs, n_l) tensor vector as the layer input, 
    # where bs is batch size.
    # The extra torch.ones added at the end here is for the bias term
    # After matrix multiplication, result gets divided by batch size for mean.
    aaT_list, abar_list = [], []
    for a_l in a:
        if a_l is None:
            aaT_list.append(None)
            abar_list.append(None)
            continue
        a1_l = torch.cat([a_l, torch.ones([a_l.shape[0], 1], device=a_l.device)], dim=1)
        aaT_list.append(torch.mm(a1_l.t(), a1_l).data / torch.tensor(a1_l.shape[0], device=a1_l.device))
        abar_list.append(a1_l.mean(dim=0))
        del a1_l
    return aaT_list, abar_list

def calc_mean_dLdS_S_aaT_abar(dLdS_blist, S_blist, aaT_blist, abar_blist):
    num_batches = len(dLdS_blist)
    num_layers = len(dLdS_blist[0])
    dLdS_layers = []
    S_layers = []
    aaT_layers = []
    abar_layers = []
    for li in range(num_layers):
        dLdS_layers.append([])
        S_layers.append([])
        aaT_layers.append([])
        abar_layers.append([])
    for bi in range(num_batches):
        for li in range(num_layers):
            dLdS_layers[li].append(dLdS_blist[bi][li])
            S_layers[li].append(S_blist[bi][li])
            aaT_layers[li].append(aaT_blist[bi][li])
            abar_layers[li].append(abar_blist[bi][li])
    dLdS_mean = []
    S_mean = []
    aaT_mean = []
    abar_mean = []
    for li in range(num_layers):
        if dLdS_layers[li][0] is None:
            dLdS_mean.append(None)
            S_mean.append(None)
            aaT_mean.append(None)
            abar_mean.append(None)
            continue
        dLdS_mean.append(torch.stack(dLdS_layers[li]).mean(dim=0))
        S_mean.append(torch.stack(S_layers[li]).mean(dim=0))
        aaT_mean.append(torch.stack(aaT_layers[li]).mean(dim=0))
        abar_mean.append(torch.stack(abar_layers[li]).mean(dim=0))

    return dLdS_mean, S_mean, aaT_mean, abar_mean


def initialize_dLdS(net_sd):
    dLdS = []
    for layerkey, layerval in net_sd.items():
        if len(layerval.shape) == 2: # Look for fully connected matrices.  Add 1 for constant bias term .
            dLdS.append(
                torch.zeros(layerval.shape[0], device=layerval.device, requires_grad=False)
            )
        elif len(layerval.shape) == 4: # Conv layers
            dLdS.append(None)
    return dLdS


def initialize_aaT(net_sd):
    aaT = []
    for layerkey, layerval in net_sd.items():
        if len(layerval.shape) == 2: # Look for fully connected matrices.  Add 1 for constant bias term .
            aaT.append(
                torch.zeros((layerval.shape[1]+1, layerval.shape[1]+1), device=layerval.device, requires_grad=False)
            )
        elif len(layerval.shape) == 4: # Conv layers
            aaT.append(None)
    return aaT

def initialize_abar(net_sd):
    abar = []
    for layerkey, layerval in net_sd.items():
        if len(layerval.shape) == 2: # Look for fully connected matrices.  Add 1 for constant bias term .
            abar.append(
                torch.zeros(layerval.shape[1]+1, device=layerval.device, requires_grad=False)
            )
        elif len(layerval.shape) == 4: # Conv layers
            abar.append(None)
    return abar

def update_grads(grad, update, scale):
    grad.add_(update * scale)
    return

def update_metrics(metrics, update, scale):
    num_layers = len(metrics)
    for li in range(num_layers):
        if metrics[li] is None:
            continue
        metrics[li].add_(update[li]*scale)
    return 

def update_Hmat(Hmat, args, epoch_idx, dLdS_curr, dLdS_last, S_curr, S_last, aaT, abar):
    num_layers = len(Hmat)
    lamb_G = lamb_A = torch.sqrt(torch.tensor(float(args.kronecker_lambda), device=args.device))
    mu1 = torch.tensor(args.kronecker_mu1, device=args.device)
    mu2 = lamb_G
    
    if args.kronecker_bc_off == True:
        bias_correction_curr = bias_correction_last = 1.0
    else:
        bias_correction_curr = 1. - args.kronecker_beta ** (epoch_idx + 1)
        bias_correction_last = 1. - args.kronecker_beta ** epoch_idx

    for li in range(num_layers):
        if Hmat[li] == None: # conv layers
            continue

        S_diff = S_curr[li] - S_last[li]
        Hmat[li]['sg'] *= bias_correction_last
        Hmat[li]['sg'] = args.kronecker_beta * Hmat[li]['sg'] + (1. - args.kronecker_beta) * S_diff
        Hmat[li]['sg'] /= bias_correction_curr        
        dLdS_diff = dLdS_curr[li] - dLdS_last[li]
        Hmat[li]['yg'] *= bias_correction_last
        Hmat[li]['yg'] = args.kronecker_beta * Hmat[li]['yg'] + (1. - args.kronecker_beta) * dLdS_diff
        Hmat[li]['yg'] /= bias_correction_curr        
        sg_tilde, yg_tilde = double_damp_per_layer(mu1, mu2, Hmat[li]['sg'], Hmat[li]['yg'], Hmat[li]['Hg'])
        Hmat[li]['Hg'] = BFGS(Hmat[li]['Hg'], sg_tilde.view(-1), yg_tilde.view(-1))

        Hmat[li]['A'] *= bias_correction_last
        Hmat[li]['A'] = args.kronecker_beta * Hmat[li]['A'] + (1. - args.kronecker_beta) * aaT[li]
        Hmat[li]['A'] /= bias_correction_curr        
        A_LM = Hmat[li]['A'] + lamb_A * torch.eye(Hmat[li]['A'].shape[0], device=args.device)
        sa = torch.mm(Hmat[li]['Ha'], abar[li].view(-1, 1))
        ya = torch.mm(A_LM, sa)
        Hmat[li]['Ha'] = BFGS(Hmat[li]['Ha'], sa.view(-1), ya.view(-1))

    return Hmat


def double_damp_per_layer(mu1, mu2, s, y, H):
    sTy = s.dot(y)
    Hy = torch.mm(H, y.view(-1, 1)).view(-1)
    yTHy = y.dot(Hy)
    if sTy < mu1 * yTHy:
        theta1_num = (1. - mu1) * yTHy
        theta1_den = yTHy - sTy
        theta1 = theta1_num.div(theta1_den)
    else:
        theta1 = 1.0
    s_tilde = theta1 * s + (1. - theta1) * Hy

    stTy = s_tilde.dot(y)
    stTst = s_tilde.dot(s_tilde)
    if stTy < mu2 * stTst:
        theta2_num = (1. - mu2) * stTst
        theta2_den = stTst - stTy
        theta2 = theta2_num.div(theta2_den)
    else:
        theta2 = 1.0
    y_tilde = theta2 * y + (1. - theta2) * s_tilde

    return s_tilde, y_tilde



def BFGS(H, s, y):
    sTy = s.dot(y)
    ssT = torch.mm(s.view(-1, 1), s.view(1, -1))
    Hy = torch.mm(H, y.view(-1, 1)).view(-1)
    yTHy = y.dot(Hy)
    HysT = torch.mm(Hy.view(-1, 1), s.view(1, -1))
    syT = torch.mm(s.view(-1, 1), y.view(1, -1))
    syTH = torch.mm(syT, H)

    H_new = H + (sTy + yTHy) * ssT / sTy / sTy - (HysT + syTH) / sTy
    return H_new



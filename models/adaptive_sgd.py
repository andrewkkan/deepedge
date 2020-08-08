import torch
from torch.optim import Optimizer
from models.sdlbfgs_fed import gather_flat_params, gather_flat_states, add_states

from IPython import embed


class Adaptive_SGD(Optimizer):
    """PyTorch Implementation of SdLBFGS algorithm [1].

    Code is adopted from LBFGS in PyTorch and modified by
    Huidong Liu (h.d.liew@gmail.com) and Yingkai Li (yingkaili2023@u.northwestern.edu)

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Arguments:
        lr (float): learning rate (default: 1).
        lr_decay (bool): whether to perform learning rate decay (default: True).
        weight_decay (float): weight decay (default: 0).
        max_iter (int): maximal number of iterations per optimization step
            (default: 1).
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).

    [1] Wang, Xiao, et al. "Stochastic quasi-Newton methods for nonconvex stochastic optimization."
    SIAM Journal on Optimization 27.2 (2017): 927-956.
    """

    def __init__(self, net, lr_server_gd=0.5, lr_device=0.1, E_l=1.0, nD=600., Bs=50., adaptive_mode=0, tau=0.0, beta1=0.9, beta2=0.99):

        defaults = dict(lr_server_gd=lr_server_gd, lr_device=lr_device, E_l=E_l, nD=nD, Bs=Bs, adaptive_mode=adaptive_mode, tau=tau, beta1=beta1, beta2=beta2)
        super(Adaptive_SGD, self).__init__(net.parameters(), defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Adaptive_SGD doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self._lr_server_gd = lr_server_gd
        self._lr_device = lr_device
        self._net = net
        self._E_l = E_l
        self._nD = nD
        self._Bs = Bs
        self._adaptive_mode = adaptive_mode
        self._tau = tau
        self._beta1 = beta1
        self._beta2 = beta2

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    def _add_other_states(self, flat_deltos):
        sd = self._net.state_dict()
        pd = dict(self._net.named_parameters())
        offset = 0
        for sdk in sd.keys():
            if sdk not in pd.keys():
                numel = sd[sdk].numel()
                if 'Long' in sd[sdk].type():
                    sd[sdk] += flat_deltos[offset:offset + numel].view_as(sd[sdk]).long()
                else:
                    sd[sdk] += flat_deltos[offset:offset + numel].view_as(sd[sdk])
                offset += numel

    def step(
            self, 
            # closure
            flat_deltw_list,
            flat_deltos_list,
        ):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        beta1 = group['beta1']
        beta2 = group['beta2']
        tau = group['tau']
        lr = group['lr_server_gd']
        mode = group['adaptive_mode']

        state = self.state['global_state']
        state.setdefault('n_iter', 0)

        if flat_deltos_list:
            flat_deltos = torch.stack(flat_deltos_list).mean(dim=0)
            self._add_other_states(flat_deltos)

        flat_deltw = torch.stack(flat_deltw_list).mean(dim=0)
        prev_flat_deltw = state.get('prev_flat_deltw')
        if prev_flat_deltw is not None:
            flat_deltw = beta1 * prev_flat_deltw + (1.0 - beta1) * flat_deltw

        prev_flat_v = state.get('prev_flat_v')
        if prev_flat_v is None:
            prev_flat_v = torch.zeros_like(flat_deltw).to(flat_deltw.device)

        delt2 = flat_deltw.pow(2.0)
        if mode == 0:
            # FedAdaGrad
            flat_v = prev_flat_v + delt2
        elif mode == 1:
            # FedYogi
            flat_v = prev_flat_v - (1.0 - beta2) * delt2 * torch.sign(prev_flat_v - delt2)
        elif mode == 2:
            # FedAdam
            flat_v = beta2 * prev_flat_v + (1.0 - beta2) * delt2

        descent = flat_deltw / (torch.sqrt(flat_v) + tau)

        # This step updates the global model with plain-vanilla device-update averaging 
        self._add_grad(lr, descent)

        if prev_flat_deltw is None:
            prev_flat_deltw = flat_deltw.clone()
        else:
            prev_flat_deltw.copy_(flat_deltw)

        if prev_flat_v is None:
            prev_flat_v = flat_v.clone()
        else:
            prev_flat_v.copy_(flat_v)

        state['prev_flat_deltw'] = prev_flat_deltw
        state['prev_flat_v'] = prev_flat_v
        state['n_iter'] += 1

        return 

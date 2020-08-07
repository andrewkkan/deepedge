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

    def __init__(self, net, lr_server_gd=0.5, lr_device=0.1, E_l=1.0, nD=600., Bs=50., adaptive_mode=0):

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
        state = self.state['global_state']
        state.setdefault('n_iter', 0)

        if flat_deltos_list:
            flat_deltos = torch.stack(flat_deltos_list).mean(dim=0)
            self._add_other_states(flat_deltos)

        flat_deltw = torch.stack(flat_deltw_list).mean(dim=0)

        if self._adaptive_mode == 0:
            # FedAdaGrad
            pass
        elif self._adaptive_mode == 1:
            # FedYogi
            pass
        elif self._adaptive_mode == 2:
            # FedAdam
            pass

        # This step updates the global model with plain-vanilla device-update averaging 
        self._add_grad(1.0, flat_deltw)
            
        flat_grad = -flat_deltw / self._lr_device / self._E_l / (self._nD / self._Bs)
        abs_grad_sum = flat_grad.abs().sum()


        prev_flat_grad = state.get('prev_flat_grad')


        if prev_flat_grad is None:
            prev_flat_grad = flat_grad.clone()
        else:
            prev_flat_grad.copy_(flat_grad)

        state['prev_flat_grad'] = prev_flat_grad
        state['n_iter'] += 1

        return 

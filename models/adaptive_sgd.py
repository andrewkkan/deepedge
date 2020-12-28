import torch
from functools import reduce
from torch.optim import Optimizer
from utils.util_model import gather_flat_params, gather_flat_states, add_states
import numpy as np

from IPython import embed


class Adaptive_SGD(Optimizer):
    """Code adapted from sdlbfgs_fed.py
    """

    def __init__(self, net, lr_server_gd=0.5, server_opt_mode=0, tau=0.0, beta1=0.9, beta2=0.99, bias_correction=False):

        defaults = dict(lr_server_gd=lr_server_gd, server_opt_mode=server_opt_mode, tau=tau, beta1=beta1, beta2=beta2)
        super(Adaptive_SGD, self).__init__(net.parameters(), defaults)

        if len(self.param_groups) != 1:
            raise ValueError("Adaptive_SGD doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self.embed = False
        self._bias_correction = bias_correction

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    @torch.no_grad()
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

    @torch.no_grad()
    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data.add_(step_size, update[offset:offset + numel].view_as(p.data))
            offset += numel
        assert offset == self._numel()

    @torch.no_grad()
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

    @torch.no_grad()
    def updateMomVals(self, mom1, mom2):
        state = self.state['global_state']
        state['prev_flat_m'] = mom1.detach().clone()
        state['prev_flat_v'] = mom2.detach().clone()

    @torch.no_grad()
    def step(
            self, 
            # closure
            flat_deltw_list,
            flat_deltos_list=None,
            nD_list=None,
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
        mode = group['server_opt_mode']

        state = self.state['global_state']
        state.setdefault('n_iter', 0)
        if state.get('step') is not None:
            state['step'] += 1
        else:
            state['step'] = 1

        if flat_deltos_list:
            flat_deltos = torch.stack(flat_deltos_list).mean(dim=0)
            self._add_other_states(flat_deltos)
        else:
            flat_deltos = None

        flat_deltw = torch.stack(flat_deltw_list).to(flat_deltw_list[0].device)
        if nD_list:
            nD_scaling = torch.tensor(nD_list).view(-1,1).to(flat_deltw_list[0].device)
        else:
            nD_scaling = torch.ones(flat_deltw.shape[0], 1).to(flat_deltw_list[0].device)
        flat_deltw = nD_scaling.mul(flat_deltw) / nD_scaling.sum().double()
        flat_deltw = flat_deltw.sum(dim=0)
        prev_flat_m = state.get('prev_flat_m')
        if prev_flat_m is None:
            flat_m = (1.0 - beta1) * flat_deltw
        else:
            flat_m = beta1 * prev_flat_m + (1.0 - beta1) * flat_deltw

        if self._bias_correction:
            bias_correction1 = 1. - beta1 ** state['step']
            bias_correction2 = 1. - beta2 ** state['step']
        else:
            bias_correction1 = 1.
            bias_correction2 = 1.

        delt2 = flat_deltw * flat_deltw
        prev_flat_v = state.get('prev_flat_v')

        if mode == 0:
            # FedAvgM (momentum only)
            descent = flat_m
            flat_v = None
        elif mode == 1:
            # FedAdaGrad
            if prev_flat_v is None:
                prev_flat_v = torch.ones_like(delt2) * (tau*tau)
            flat_v = prev_flat_v + delt2
            descent = flat_m / (torch.sqrt(flat_v)/np.sqrt(bias_correction2)  + tau)
        elif mode == 2:
            # FedYogi
            if prev_flat_v is None:
                prev_flat_v = torch.ones_like(delt2) * (tau*tau)
            flat_v = prev_flat_v - (1.0 - beta2) * delt2 * torch.sign(prev_flat_v - delt2)
            descent = flat_m / (torch.sqrt(flat_v)/np.sqrt(bias_correction2) + tau)
        elif mode == 3:
            # FedAdam
            if prev_flat_v is None:
                prev_flat_v = torch.ones_like(delt2) * (tau*tau)
            flat_v = beta2 * prev_flat_v + (1.0 - beta2) * delt2
            descent = flat_m / (torch.sqrt(flat_v)/np.sqrt(bias_correction2) + tau)

        # This step updates the global model with plain-vanilla device-update averaging 
        self._add_grad(lr / bias_correction1, -descent)

        if prev_flat_m is None:
            prev_flat_m = flat_m.clone()
        else:
            prev_flat_m.copy_(flat_m)

        if mode > 0:
            if prev_flat_v is None:
                prev_flat_v = flat_v.clone()
            else:
                prev_flat_v.copy_(flat_v)

        state['prev_flat_m'] = prev_flat_m
        state['prev_flat_v'] = prev_flat_v
        state['n_iter'] += 1

        del flat_deltw
        if flat_deltos:
            del flat_deltos
        del nD_scaling
        del delt2
        del flat_m
        if flat_v is not None:
            del flat_v
        del descent
        del prev_flat_m
        del prev_flat_v

        return 

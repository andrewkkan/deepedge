import torch
from functools import reduce
from torch.optim import Optimizer
from IPython import embed

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

class SdLBFGS_FedLiSA(Optimizer):
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

    def __init__(self, net, lr_server_qn=0.5, lr_server_gd=1.0, lr_decay=False, weight_decay=0, max_iter=1, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-9, history_size=100,
                 line_search_fn=None, lr_device=0.1, E_l=1.0, nD=600., Bs=50., 
                 opt_mode=0, vr_mode=0, max_qndn=1.0):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(lr_server_qn=lr_server_qn, lr_decay=lr_decay, weight_decay=weight_decay, max_iter=max_iter,
                        max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size, line_search_fn=line_search_fn)
        super(SdLBFGS_FedLiSA, self).__init__(net.parameters(), defaults)

        if len(self.param_groups) != 1:
            raise ValueError("SdLBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None
        self._lr_server_qn = lr_server_qn
        self._lr_server_gd = lr_server_gd
        self._lr_device = lr_device
        self._net = net
        self._E_l = E_l
        self._nD = nD
        self._Bs = Bs
        self._opt_mode = opt_mode
        self._vr_mode = vr_mode
        self._max_qndn = max_qndn

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

    def _add_weight_decay(self, weight_decay, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            update[offset:offset + numel].add_(weight_decay, p.data.view(-1))
            offset += numel
        return update

    def _calc_beta(self, s, y, eta, theta):
        # Compute products
        u = s.dot(s)
        v = s.dot(y)
        w = y.dot(y)
        betas = torch.FloatTensor([0,0])
        # Compute lower bound
        if eta > v.div(u):
            betas[0] = (eta*u-v)/(u-v)
        vv = betas[0]*s + (1-betas[0])*y
        if betas[0] > 0 and vv.dot(vv) > theta*s.dot(vv):
            betas[0] = 1.
        # Compute upper bound
        if w / v > theta:
            a = (u-2*v+w)
            b = (2*v-2*w-theta*u+theta*v)
            c = (w-theta*v)
            sqroot = (b.square() - 4.*a*c)
            if sqroot >= 0.:
                betas[1] = (-b -sqroot.pow(0.5)) / 2. / a
            else:
                betas[1] = -b / 2. / a
        if betas[0] >= betas[1]:
            return betas[0]
        else:
            return betas[1]

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

        if flat_deltos_list:
            flat_deltos = torch.stack(flat_deltos_list).mean(dim=0)
            self._add_other_states(flat_deltos)

        flat_deltw = torch.stack(flat_deltw_list).mean(dim=0)

        if self._opt_mode != 1:
            # This step updates the global model with plain-vanilla device-update averaging 
            self._add_grad(self._lr_server_gd, flat_deltw)
            if self._opt_mode == 0:
                return

        group = self.param_groups[0]
        lr_server_qn = group['lr_server_qn']
        lr_decay = group['lr_decay']
        weight_decay = group['weight_decay']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        state = self.state['global_state']
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        # orig_loss = closure()
        # loss = orig_loss.item()
        current_evals = 1
        state['func_evals'] += 1
            
        flat_grad = -flat_deltw / self._lr_device / self._E_l / (self._nD / self._Bs)
        abs_grad_sum = flat_grad.abs().sum()

        if abs_grad_sum <= tolerance_grad:
            print("SdBFGS optim exited prematurely!")
            return # loss

        # variables cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        # prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if lr_decay:
                t = lr_server_qn / (state['n_iter'] ** 0.5)
                # a0 = 1./16.
                # a1 = 0.
                # t = 1/(a0 + state['n_iter']*a1)
            else:
                if state['n_iter'] == 1:
                    t = min(1., 1. / abs_grad_sum) * lr_server_qn
                else:
                    t = lr_server_qn

            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                H_diag = 1
            else:
                # do Sdlbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t) #+ flat_deltw * self._lr_g
                ys = y.dot(s)  # y*s

                # update scale of initial Hessian approximation
                gamma = 1
                H_diag = 1 / gamma
                sT_H_inv_s = gamma * s.dot(s)

                if ys < 0.25 * sT_H_inv_s:
                    theta = 0.75 * sT_H_inv_s / (sT_H_inv_s - ys)
                else:
                    theta = 1
                y_bar = theta * y + (1 - theta) * gamma * s

                # eta = 0.25
                # theta = 4.
                # beta = self._calc_beta(s,t*y,eta,theta)
                # y_bar = beta * s + (1. - beta) * t * y

                # updating memory
                if len(old_dirs) == history_size:
                    # shift history by one (limited-memory)
                    old_dirs.pop(0)
                    old_stps.pop(0)

                # store new direction/step
                old_dirs.append(s)
                old_stps.append(y_bar)

                # compute the approximate (SdL-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'ro' not in state:
                    state['ro'] = [None] * history_size
                    state['al'] = [None] * history_size
                ro = state['ro']
                al = state['al']

                for i in range(num_old):
                    ro[i] = 1. / old_stps[i].dot(old_dirs[i])

                # iteration in SdL-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_dirs[i].dot(q) * ro[i]
                    q.add_(-al[i], old_stps[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_stps[i].dot(r) * ro[i]
                    r.add_(al[i] - be_i, old_dirs[i])

                #print(theta, t, d.norm(), q.norm(), y.norm(), s.norm(), ys.norm(), flat_grad.norm())

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone()
            else:
                prev_flat_grad.copy_(flat_grad)
            # prev_loss = loss

            ############################################################
            # compute step size
            ############################################################
            # reset initial guess for step size
            if weight_decay > 0:
                d = self._add_weight_decay(weight_decay, d)

            dnorm = d.norm()
            if dnorm > self._max_qndn/t:
                d = d / d.norm() / t * self._max_qndn 

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                raise RuntimeError("line search function is not supported yet")
            else:
                if self._opt_mode == 1 or self._opt_mode == 2:
                    # no line search, simply move with fixed-step 
                    self._add_grad(t, d)

                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    # loss = closure().item()
                    # flat_grad = self._gather_flat_grad()
                    # abs_grad_sum = flat_grad.abs().sum()
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                # print('Condition 1 not met')
                break

            if current_evals >= max_eval:
                print('Condition 2 not met')
                break

            if abs_grad_sum <= tolerance_grad:
                print('Condition 3 not met')
                break

            if gtd > -tolerance_change:
                print('Condition 4 not met')
                break

            if d.mul(t).abs_().sum() <= tolerance_change:
                print('Condition 5 not met')
                break

            # if abs(loss - prev_loss) < tolerance_change:
            #     break

        state['d'] = d 
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        # state['prev_loss'] = prev_loss

        return t*d

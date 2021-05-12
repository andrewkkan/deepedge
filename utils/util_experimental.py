import torch
import numpy as np
import copy
from utils.util_model import gather_flat_params, gather_flat_params_with_grad, gather_flat_other_states, gather_flat_grad, gather_flat_states, add_states

def simple_linesearch(flat_descent, net, flat_grad, loss_func, images_labels, task, curr_loss):
    (images, labels) = images_labels
    num_trials = 8
    half_index = int(num_trials/2.)
    depth_level = 3
    dwell_offset = 0.0
    for dl in range(depth_level):
        loss_trials = []
        for nt in range(num_trials):
            search_net = copy.deepcopy(net)
            scale = float(nt+0.5) / float(num_trials) / np.power(2.0,dl) + dwell_offset
            add_states(search_net, -flat_descent * scale)
            nn_outputs, _, _ = search_net(images)
            if task == 'AutoEnc':
                loss = loss_func(nn_outputs, images) / images.shape[-1] / images.shape[-2] / images.shape[-3]
            else:
                loss = loss_func(nn_outputs, labels) 
            loss_trials.append(loss.item())
            del search_net
        losses_np = np.array(loss_trials)
        loss_sum1 = losses_np[0:half_index].sum()
        loss_sum2 = losses_np[half_index:].sum()
        if loss_sum1 >= loss_sum2:
            dwell_offset += np.power(2.0, -(dl+1))
    if loss_sum1 >= loss_sum2:
        final_loss_avg = loss_sum2 / float(half_index)
    else:
        final_loss_avg = loss_sum1 / float(half_index)
    if final_loss_avg <= curr_loss:
        return flat_descent * (np.power(2.0, -(dl+2)) + dwell_offset)
    else:
        return torch.zeros_like(flat_descent)

def weak_wolfe_linesearch():
    # https://github.com/hjmshi/PyTorch-LBFGS/blob/master/functions/LBFGS.py
    # load options
    if options:
        if 'closure' not in options.keys():
            raise(ValueError('closure option not specified.'))
        else:
            closure = options['closure']

        if 'current_loss' not in options.keys():
            F_k = closure()
            closure_eval += 1
        else:
            F_k = options['current_loss']

        if 'gtd' not in options.keys():
            gtd = g_Sk.dot(d)
        else:
            gtd = options['gtd']

        if 'eta' not in options.keys():
            eta = 2
        elif options['eta'] <= 1:
            raise(ValueError('Invalid eta; must be greater than 1.'))
        else:
            eta = options['eta']

        if 'c1' not in options.keys():
            c1 = 1e-4
        elif options['c1'] >= 1 or options['c1'] <= 0:
            raise(ValueError('Invalid c1; must be strictly between 0 and 1.'))
        else:
            c1 = options['c1']

        if 'c2' not in options.keys():
            c2 = 0.9
        elif options['c2'] >= 1 or options['c2'] <= 0:
            raise(ValueError('Invalid c2; must be strictly between 0 and 1.'))
        elif options['c2'] <= c1:
            raise(ValueError('Invalid c2; must be strictly larger than c1.'))
        else:
            c2 = options['c2']

        if 'max_ls' not in options.keys():
            max_ls = 10
        elif options['max_ls'] <= 0:
            raise(ValueError('Invalid max_ls; must be positive.'))
        else:
            max_ls = options['max_ls']

        if 'interpolate' not in options.keys():
            interpolate = True
        else:
            interpolate = options['interpolate']

        if 'inplace' not in options.keys():
            inplace = True
        else:
            inplace = options['inplace']
            
        if 'ls_debug' not in options.keys():
            ls_debug = False
        else:
            ls_debug = options['ls_debug']

    else:
        raise(ValueError('Options are not specified; need closure evaluating function.'))

    # initialize counters
    ls_step = 0
    grad_eval = 0 # tracks gradient evaluations
    t_prev = 0 # old steplength

    # initialize bracketing variables and flag
    alpha = 0
    beta = float('Inf')
    fail = False

    # initialize values for line search
    if(interpolate):
        F_a = F_k
        g_a = gtd

        if(torch.cuda.is_available()):
            F_b = torch.tensor(np.nan, dtype=dtype).cuda()
            g_b = torch.tensor(np.nan, dtype=dtype).cuda()
        else:
            F_b = torch.tensor(np.nan, dtype=dtype)
            g_b = torch.tensor(np.nan, dtype=dtype)

    # begin print for debug mode
    if ls_debug:
        print('==================================== Begin Wolfe line search ====================================')
        print('F(x): %.8e  g*d: %.8e' % (F_k, gtd))

    # check if search direction is descent direction
    if gtd >= 0:
        desc_dir = False
        if debug:
            print('Not a descent direction!')
    else:
        desc_dir = True

    # store values if not in-place
    if not inplace:
        current_params = self._copy_params()

    # update and evaluate at new point
    self._add_update(t, d)
    F_new = closure()
    closure_eval += 1

    # main loop
    while True:

        # check if maximum number of line search steps have been reached
        if ls_step >= max_ls:
            if inplace:
                self._add_update(-t, d)
            else:
                self._load_params(current_params)

            t = 0
            F_new = closure()
            F_new.backward()
            g_new = self._gather_flat_grad()
            closure_eval += 1
            grad_eval += 1
            fail = True
            break

        # print info if debugging
        if ls_debug:
            print('LS Step: %d  t: %.8e  alpha: %.8e  beta: %.8e' 
                  % (ls_step, t, alpha, beta))
            print('Armijo:  F(x+td): %.8e  F-c1*t*g*d: %.8e  F(x): %.8e'
                  % (F_new, F_k + c1 * t * gtd, F_k))

        # check Armijo condition
        if F_new > F_k + c1 * t * gtd:

            # set upper bound
            beta = t
            t_prev = t

            # update interpolation quantities
            if interpolate:
                F_b = F_new
                if torch.cuda.is_available():
                    g_b = torch.tensor(np.nan, dtype=dtype).cuda()
                else:
                    g_b = torch.tensor(np.nan, dtype=dtype)

        else:

            # compute gradient
            F_new.backward()
            g_new = self._gather_flat_grad()
            grad_eval += 1
            gtd_new = g_new.dot(d)
            
            # print info if debugging
            if ls_debug:
                print('Wolfe: g(x+td)*d: %.8e  c2*g*d: %.8e  gtd: %.8e'
                      % (gtd_new, c2 * gtd, gtd))

            # check curvature condition
            if gtd_new < c2 * gtd:

                # set lower bound
                alpha = t
                t_prev = t

                # update interpolation quantities
                if interpolate:
                    F_a = F_new
                    g_a = gtd_new

            else:
                break

        # compute new steplength

        # if first step or not interpolating, then bisect or multiply by factor
        if not interpolate or not is_legal(F_b):
            if beta == float('Inf'):
                t = eta*t
            else:
                t = (alpha + beta)/2.0

        # otherwise interpolate between a and b
        else:
            t = polyinterp(np.array([[alpha, F_a.item(), g_a.item()], [beta, F_b.item(), g_b.item()]]))

            # if values are too extreme, adjust t
            if beta == float('Inf'):
                if t > 2 * eta * t_prev:
                    t = 2 * eta * t_prev
                elif t < eta * t_prev:
                    t = eta * t_prev
            else:
                if t < alpha + 0.2 * (beta - alpha):
                    t = alpha + 0.2 * (beta - alpha)
                elif t > (beta - alpha) / 2.0:
                    t = (beta - alpha) / 2.0

            # if we obtain nonsensical value from interpolation
            if t <= 0:
                t = (beta - alpha) / 2.0

        # update parameters
        if inplace:
            self._add_update(t - t_prev, d)
        else:
            self._load_params(current_params)
            self._add_update(t, d)

        # evaluate closure
        F_new = closure()
        closure_eval += 1
        ls_step += 1

    # store Bs
    if Bs is None:
        Bs = (g_Sk.mul(-t)).clone()
    else:
        Bs.copy_(g_Sk.mul(-t))
        
    # print final steplength
    if ls_debug:
        print('Final Steplength:', t)
        print('===================================== End Wolfe line search =====================================')

    state['d'] = d
    state['prev_flat_grad'] = prev_flat_grad
    state['t'] = t
    state['Bs'] = Bs
    state['fail'] = fail

    return F_new, g_new, t, ls_step, closure_eval, grad_eval, desc_dir, fail

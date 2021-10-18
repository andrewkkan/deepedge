from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import numpy as np


def update_hyper_grad(
        hyper_grad_vals: Dict[str, Union[float, List[float]]], 
        hyper_grad_lr_alpha: Dict[str, float], 
        hyper_grad_mom: Dict[str, Union[float, List[float]]],
        hyper_grad_agg: List[Dict[str, Union[float, List[float]]]],
        round_idx: int,
) -> None:

    for key, val in hyper_grad_vals.items():
        if type(val) == float:
            hyper_grad_update: float = 0.0
        elif type(val) == list:
            hyper_grad_update: np.array = np.zeros_like(np.asarray(val), dtype=np.float64)
        num_updates = 0
        for hg in hyper_grad_agg:
            try:
                key_val = hg[key]
            except:
                continue
            else:
                if type(key_val) == list:
                    key_val = np.asarray(key_val, dtype=np.float64)
                hyper_grad_update += key_val
                num_updates += 1
        if num_updates:
            hyper_grad_update = hyper_grad_update / float(num_updates)
            lr = hyper_grad_lr_alpha[key]['lr']
            alpha = hyper_grad_lr_alpha[key]['alpha']
            bias_correction = 1.0 - alpha ** (round_idx + 1)
            if type(hyper_grad_vals[key]) == list:
                hypergrad_val = np.asarray(hyper_grad_vals[key], dtype=np.float64) 
                hypergrad_mom = (np.asarray(hyper_grad_mom[key], dtype=np.float64) * alpha \
                    + lr * hyper_grad_update * (1.0 - alpha)) / bias_correction
                hypergrad_val += hypergrad_mom
                hypergrad_val = hypergrad_val.tolist()
            else:
                hypergrad_mom = (hyper_grad_mom[key]* alpha \
                    + lr * hyper_grad_update * (1.0 - alpha)) / bias_correction
                hypergrad_val = hyper_grad_vals[key] + hypergrad_mom
            hyper_grad_vals[key] = hypergrad_val
            hyper_grad_mom[key] = hypergrad_mom


def calculate_gradient_ref(grad_current: torch.Tensor, grad_w: torch.Tensor, momentum_alpha: float, epoch_idx: int,
) -> torch.Tensor:

    bias_correction = 1.0 - momentum_alpha ** (epoch_idx + 1)
    grad_with_momentum = (grad_current * momentum_alpha + grad_w * (1.0 - momentum_alpha)) / bias_correction
    return grad_with_momentum



def calcuate_lr_global(lr_global_current: float, hlr: float, desc_ref: torch.Tensor, delt_w: torch.Tensor,
) -> float:

    cos_similarity: float = (desc_ref * delt_w).sum()
    print(f"cos_similarity = {cos_similarity} \n")
    lr_global_new = lr_global_current + hlr * cos_similarity
    return lr_global_new




from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import numpy as np


def update_hyper_grad(
        hyper_grad_vals: Dict[str, Union[float, List[float]]], 
        hyper_grad_lr_alpha: Dict[str, float], 
        hyper_grad_mom: Dict[str, Union[float, List[float]]],
        hyper_grad_agg: List[Dict[str, Union[float, List[float]]]],
        round_idx: int,
        sync_idx: int,
) -> None:

    for key, val in hyper_grad_vals.items():
        hyper_grad_update_float: float = 0.0
        hyper_grad_update_array: np.array = np.zeros_like(np.asarray(val), dtype=np.float64)
        num_updates_float = 0
        num_updates_array = 0
        for hg in hyper_grad_agg:
            try:
                key_val = hg[key]
            except:
                continue
            else:
                if type(key_val) == list:
                    key_val = np.asarray(key_val, dtype=np.float64)
                    hyper_grad_update_array += key_val
                    num_updates_array += 1
                elif type(key_val) == float:
                    hyper_grad_update_float += key_val
                    num_updates_float += 1
        if num_updates_array > 0 or num_updates_float > 0:
            lr = hyper_grad_lr_alpha[key]['lr']
            alpha = hyper_grad_lr_alpha[key]['alpha']
            bias_correction = 1.0 - alpha ** (round_idx + 1)
            if num_updates_array:  # for now just assume either float or array update, not both
                hyper_grad_update_array = hyper_grad_update_array / float(num_updates_array)
                hypergrad_val: np.array = np.asarray(hyper_grad_vals[key], dtype=np.float64) 
                hypergrad_mom: np.array = (np.asarray(hyper_grad_mom[key], dtype=np.float64) * alpha \
                    + hyper_grad_update * (1.0 - alpha)) / bias_correction
                hypergrad_val: np.array = hypergrad_val + lr * hypergrad_mom
                hypergrad_val: list = hypergrad_val.tolist()
                hypergrad_mom: list = hypergrad_mom.tolist()
            elif num_updates_float:  # for now just assume either float or array update, not both
                hyper_grad_update_float = hyper_grad_update_float / float(num_updates_float)
                if type(hyper_grad_vals[key]) == list:
                    hypergrad_val:float = hyper_grad_vals[key][sync_idx]
                    hypergrad_mom:float = hyper_grad_mom[key][sync_idx]
                    hypergrad_mom:float = (hypergrad_mom* alpha \
                        + hyper_grad_update_float * (1.0 - alpha)) / bias_correction
                    hypergrad_val:float = hypergrad_val + lr * hypergrad_mom
                    hypergrad_val_list = hyper_grad_vals[key]
                    hypergrad_val_list[sync_idx] = hypergrad_val
                    hypergrad_val: list = hypergrad_val_list
                    hypergrad_mom_list = hyper_grad_mom[key]
                    hypergrad_mom_list[sync_idx] = hypergrad_mom
                    hypergrad_mom: list = hypergrad_mom_list                    
                else:
                    hypergrad_mom:float = (hyper_grad_mom[key]* alpha \
                        + hyper_grad_update * (1.0 - alpha)) / bias_correction
                    hypergrad_val:float = hyper_grad_vals[key] + lr * hypergrad_mom

            hyper_grad_vals[key]: Union[float, List[float]] = hypergrad_val
            hyper_grad_mom[key]: Union[float, List[float]] = hypergrad_mom


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




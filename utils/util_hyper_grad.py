from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import numpy as np


def update_hyper_grad(hyper_grad_vals: Dict[str, Union[float, List[float]]], hyper_grad_lr: Dict[str, float], hyper_grad_agg: List[Dict[str, Union[float, List[float]]]],
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
            if type(hyper_grad_vals[key]) == list:
                hypergrad_val = np.asarray(hyper_grad_vals[key], dtype=np.float64) + hyper_grad_lr[key] * hyper_grad_update
                hypergrad_val = hypergrad_val.tolist()
            else:
                hypergrad_val = hyper_grad_vals[key] + hyper_grad_lr[key] * hyper_grad_update
            hyper_grad_vals[key] = hypergrad_val



def calculate_global_descent(desc_current: torch.Tensor, delt_w: torch.Tensor, momentum_alpha: float, epoch_idx: int,
) -> torch.Tensor:

    bias_correction = 1.0 - momentum_alpha ** (epoch_idx + 1)
    desc_with_momentum = (desc_current * momentum_alpha + delt_w * (1.0 - momentum_alpha)) / bias_correction
    return desc_with_momentum



def calcuate_lr_global(lr_global_current: float, hlr: float, desc_ref: torch.Tensor, delt_w: torch.Tensor,
) -> float:

    cos_similarity: float = (desc_ref * delt_w).sum()
    print(f"cos_similarity = {cos_similarity} \n")
    lr_global_new = lr_global_current + hlr * cos_similarity
    return lr_global_new




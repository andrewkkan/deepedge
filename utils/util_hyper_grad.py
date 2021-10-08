from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import numpy as np


def update_hyper_grad(hyper_grad_vals: Dict[str, float], hyper_grad_lr: Dict[str, float], hyper_grad_agg: List[Dict[str, float]]) -> None:
    num_updates = len(hyper_grad_agg)
    if num_updates < 1:
        return
    for key, val in hyper_grad_vals.items():
        hyper_grad_update: float = 0.0
        for hg in hyper_grad_agg:
            hyper_grad_update += hg[key]
        hyper_grad_update = hyper_grad_update / float(num_updates)
        hyper_grad_vals[key] += hyper_grad_lr[key] * hyper_grad_update
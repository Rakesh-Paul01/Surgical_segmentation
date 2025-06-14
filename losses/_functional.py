from typing import Optional

import math
import numpy as np

import torch
import torch.linalg as LA
import torch.nn.functional as F

__all__ = [
    "focal_loss_with_logits",
    "softmax_focal_loss_with_logits",
    "soft_jaccard_score",
    "soft_dice_score",
    "wing_loss"
]

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


def soft_tversky_score(
        output: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        beta: float,
        smooth: float = 0.0,
        eps: float= 1e-7,
        dims = None
) -> torch.Tensor:
    """Tversky score

    Args:
        output (torch.Tensor): ouput from the model
        target (torch.Tensor): ground truth values
        alpha (float): alpha value
        beta (float): beta value
        smooth (float, optional): smooth values. Defaults to 0.0.
        eps (float, optional): eps value. Defaults to 1e-7.
        dims (_type_, optional): dimensions. Defaults to None.

    Returns:
        torch.Tensor: tversky loss 

    References:
        https://arxiv.org/pdf/2302.05666
        https://arxiv.org/pdf/2303.16296
    """
    
    assert output.size() == target.size()

    if dims is not None:
        output_sum = torch.sum(output, dim = dims)
        target_sum = torch.sum(target, dim=dims)
        difference = LA.vector_norm(output - target, ord =1 , dim= dims)
    else:
        output_sum = torch.sum(output)
        target_sum = torch.sum(target)
        difference = LA.vector_norm(output - target, ord =1)

    intersection = (output_sum + target_sum - difference) / 2
    fp = output_sum - intersection
    fn = target_sum - intersection

    tversky_score = (intersection + smooth) / (
        intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)

    return tversky_score

    
def soft_dice_score(
        output: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dims = None
) -> torch.Tensor:

    """Computes the dice score which is to be used on dice loss

    Reference: 
        https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/losses/_functional.py

    Returns:
        torch.Tensor: dice score
    """
    assert output.size() == target.size()
    dice_score = soft_tversky_score(
        output=output,
        target=target,
        alpha=1,
        beta=1.0,
        smooth=smooth,
        eps=eps,
        dims=dims
    )
    return dice_score
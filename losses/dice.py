from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from ._functional import soft_dice_score, to_tensor

__all__ = ['DiceLoss']

class DiceLoss(_Loss):
    def __init__(self, 
                 mode:str,
                 classes: Optional[List[int]] = None,
                 log_loss: bool = False,
                 from_logits: bool = True,
                 smooth: float = 0.0,
                 ignore_index: Optional[int] = None,
                 eps: float = 1e-7
                 ):
        super(DiceLoss, self).__init__()
        assert mode in {"binary", "Multiclass", "multilabel"}
        self.mode = mode
        if classes is not None:
            assert mode != "binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, predictions: torch.Tensor, ground_truth: torch.Tensor):
        assert ground_truth.size(0) == predictions.size(0)

        # this is for numerical stability
        if self.mode == 'multiclass':
            if self.from_logits:
                predictions = F.logsigmoid(predictions).exp()

        batch_size = ground_truth.size(0)
        num_classes = predictions.size(1)
        dims = (0, 2)

        if self.mode == 'binary':
            ground_truth = ground_truth.view(batch_size, 1, -1)
            predictions = predictions.view(batch_size, 1, -1)

            if self.ignore_index is not None:
                mask = ground_truth != self.ignore_index
                ground_truth = ground_truth * mask
                predictions = predictions * mask
        
        if self.mode == 'multiclass':
            ground_truth = ground_truth.view(batch_size, -1)
            predictions = predictions.view(batch_size, num_classes, -1)

            if self.ignore_index is not None:
                mask = ground_truth != self.ignore_index
                predictions = predictions * mask.unsqueeze(1)

                predictions = F.one_hot(
                    (ground_truth * mask).to(torch.long), num_classes
                )
                ground_truth = ground_truth.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                ground_truth = F.one_hot(ground_truth, num_classes)
                ground_truth = ground_truth.permute(0,2,1)

        if self.mode == 'multilabel':
            ground_truth = ground_truth.view(batch_size, num_classes, -1)
            predictions = predictions.view(batch_size, num_classes, -1)

            if self.ignore_index is not None:
                mask = ground_truth != self.ignore_index
                ground_truth = ground_truth * mask
                predictions = predictions * mask
            
        scores = self.compute_score(
            predictions, ground_truth.type_as(predictions), smooth = self.smooth, eps = self.eps, dims= dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores
        
        mask = ground_truth.sum(dims)> 0

    
    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth= 0.0, eps= 1e-7, dims= None) -> torch.Tensor:
        return soft_dice_score(output=output, target=target, smooth=smooth, eps=eps, dims=dims)
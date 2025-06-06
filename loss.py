from typing import List, Tuple, Dict, Optional

from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F


class DiceLoss(_Loss):
    def __init__(self, mode: str, 
                 classes: Optional[List[int]] = None, 
                 log_loss: bool = False,
                 from_logits: bool = True,
                 smooth: float = 0.0,
                 ignore_index: Optional[int] = None,
                 eps: float= 1e-7
                 ):
        super().__init__()

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true = torch.Tensor) -> torch.Tensor:
        assert y_true.shape == y_pred.shape

        if self.from_logits:
            y_pred = y_pred.log_softmax(dim=1).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0,2)
        
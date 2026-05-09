"""
src/ml/loss.py

Asymmetric Focal Loss for multi-label classification with extreme class imbalance.
Implements Ridnik et al. (ICCV 2021): "Asymmetric Loss For Multi-Label Classification"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric focal loss for multi-label classification with extreme imbalance.

    Uses separate focal exponents for positive and negative samples to handle
    the case where negatives vastly outnumber positives.

    Args:
        gamma_pos (float): Focal exponent for positive samples (typically 0.0).
        gamma_neg (float): Focal exponent for negative samples (typically 2.0-4.0).
        clip (float): Probability clipping for numerical stability [default 0.05].
    """

    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric focal loss.

        Args:
            logits: (B, C) or (B, C, ...) raw model output (pre-sigmoid)
            labels: (B, C) or (B, C, ...) binary 0/1 targets

        Returns:
            scalar loss (mean reduction across all elements)
        """
        # Compute sigmoid(logits) and clip for numerical stability
        probs = torch.sigmoid(logits)
        probs = probs.clamp(min=self.clip, max=1 - self.clip)

        # Focal loss: separate gamma for positive and negative samples
        # Positive samples (labels=1): (1 - p)^gamma_pos * log(p)
        # Negative samples (labels=0): p^gamma_neg * log(1 - p)
        pos = labels * (1 - probs) ** self.gamma_pos * torch.log(probs)
        neg = (1 - labels) * probs ** self.gamma_neg * torch.log(1 - probs)

        loss = -(pos + neg).mean()
        return loss

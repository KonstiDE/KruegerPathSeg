import torch
import torch.nn as nn


class DiceSBLoss(nn.Module):
    def __init__(self, beta=0.7):
        super(DiceSBLoss, self).__init__()

        self.beta = beta

    def forward(self, pred, target):
        term1 = self.beta * pred + (1 - self.beta) * target

        numerator = 1 + torch.sum(2 * term1 * target)
        denominator = 1 + torch.sum(term1 ** 2 + target ** 2)

        return 1 - (numerator / denominator)

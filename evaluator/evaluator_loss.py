import torch
import torch.nn as nn


class EvaluatorLoss(nn.Module):

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, true_outputs, generator_outputs, other_outputs):
        result = torch.log(true_outputs) + self.alpha + torch.log(1 - generator_outputs) + self.beta * torch.log(
            1 - other_outputs)
        result = -result
        return result

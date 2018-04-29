import torch
import torch.nn as nn


class RLLoss(nn.Module):

    def forward(self, rewards, props):
        loss = rewards * props
        loss = -torch.sum(loss)
        return loss

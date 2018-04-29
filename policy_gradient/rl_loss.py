import torch
from torch.autograd import Function


class RLLoss(Function):

    @staticmethod
    def forward(ctx, input):
        grads, final_reward = input
        ctx.save_for_backward(grads)
        return final_reward.mean()

    @staticmethod
    def backward(ctx, grad_output):
        grads, = ctx.saved_tensors
        grads = torch.mean(grads, dim=0)
        return grads

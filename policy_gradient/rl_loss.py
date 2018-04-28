import torch
from torch.autograd import Function


class RLLoss(Function):

    @staticmethod
    def forward(ctx, input):
        grads, rewards = input
        ctx.save_for_backward(input)
        return rewards[-1].view(-1).mean()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grads, rewards = input
        temp = grads * rewards
        temp = torch.mean(temp, dim=0)
        return temp

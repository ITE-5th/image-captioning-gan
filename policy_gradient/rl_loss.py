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
        grads, = ctx.saved_tensors  # (batch size, max sentence length, vocab size)
        # for one step
        grads = torch.mean(grads, dim=1)  # (batch_size, 1, vocab_size)
        grads = grads.squeeze(1)  # ()
        # for batches
        grads = torch.mean(grads, dim=0)  # (vocab size)
        grads = grads.view(-1)  # (vocab size)
        return grads

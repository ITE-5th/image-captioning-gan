# -*- coding:utf-8 -*-

import copy

import torch


class Rollout:
    """Roll-out policy"""

    def __init__(self, max_sentence_length, corpus):
        self.embed = corpus
        self.lstm = None
        self.max_sentence_length = max_sentence_length
        self.output_linear = None

    def reward(self, generated, hidden, monte_carlo_count, evaluator):
        with torch.no_grad():
            batch_size = generated.size(0)
            result = torch.zeros(batch_size, 1)
            remaining = self.max_sentence_length - generated.shape[1]
            original_hidden = hidden
            for j in range(monte_carlo_count):
                current_generated = generated
                hidden = original_hidden
                inputs = generated[:, -1].view(batch_size, 1, -1)
                for i in range(remaining):
                    _, hidden = self.lstm(inputs, hidden)
                    outputs = self.output_linear(hidden[0]).squeeze(0)
                    predicted = outputs.multinomial(1).view(batch_size, -1)
                    # embed the next inputs, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
                    inputs = self.embed.word_embeddings_from_indices(predicted.cpu().data.numpy()).unsqueeze(
                        1).cuda()
                    current_generated = torch.cat([current_generated, inputs], dim=1)
                reward = evaluator(current_generated)
                reward = reward.view(batch_size, -1)
                result += reward
            result /= monte_carlo_count
            return result

    def update(self, original_model):
        self.lstm = copy.deepcopy(original_model.lstm)
        self.output_linear = copy.deepcopy(original_model.output_linear)

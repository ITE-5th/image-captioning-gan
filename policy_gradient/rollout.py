# -*- coding:utf-8 -*-

import copy

import torch
import torch.nn.functional as F


class Rollout:
    """Roll-out policy"""

    def __init__(self, max_sentence_length, corpus):
        self.embed = corpus
        self.lstm = None
        self.max_sentence_length = max_sentence_length
        self.output_linear = None

    def reward(self, generated, image_features, hidden, monte_carlo_count, evaluator):
        with torch.no_grad():
            batch_size = generated.size(0)
            embedding_dim = generated.shape[-1]
            result = torch.zeros(batch_size, 1).cuda()
            remaining = self.max_sentence_length - generated.shape[1]
            hidden = hidden.repeat(1, 1, monte_carlo_count).view(batch_size, -1, embedding_dim)
            generated = generated.repeat(1, 1, monte_carlo_count).view(batch_size, -1, embedding_dim)
            inputs = generated[:, -1].view(batch_size, 1, -1)
            current_generated = generated
            for i in range(remaining):
                _, hidden = self.lstm(inputs, hidden)
                outputs = self.output_linear(hidden[0]).squeeze(0)
                outputs = F.softmax(outputs, -1)
                predicted = outputs.multinomial(1)
                # embed the next inputs, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
                inputs = self.embed.word_embeddings_from_indices(predicted.view(-1).cpu().data.numpy()).unsqueeze(
                    1).cuda()
                current_generated = torch.cat([current_generated, inputs], dim=1)
            image_features = image_features.repeat(1, monte_carlo_count).view(batch_size, -1)
            reward = evaluator(image_features, current_generated)
            reward = torch.tensor([reward[i:i + monte_carlo_count].sum() for i in
                                   range(0, batch_size * monte_carlo_count, monte_carlo_count)]).view(batch_size,
                                                                                                      -1).cuda()
            reward = reward.view(batch_size, -1)
            result += reward
            result /= monte_carlo_count
            return result

    def update(self, original_model):
        self.lstm = copy.deepcopy(original_model.lstm)
        self.lstm.flatten_parameters()
        self.output_linear = copy.deepcopy(original_model.output_linear)

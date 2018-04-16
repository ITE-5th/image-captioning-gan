import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal

from dataset.corpus import Corpus
from file_path_manager import FilePathManager
from misc.net_util import remove_module


class ConditionalGenerator(nn.Module):

    def __init__(self,
                 corpus: Corpus,
                 mean: torch.FloatTensor = torch.zeros(1024),
                 std: torch.FloatTensor = torch.ones(1024),
                 cnn_output_size: int = 4096,
                 input_encoding_size: int = 512,
                 max_sentence_length: int = 16,
                 num_layers: int = 1,
                 dropout: float = 0):
        super().__init__()

        self.cnn_output_size = cnn_output_size
        self.input_encoding_size = input_encoding_size
        self.max_sentence_length = max_sentence_length
        self.embed = corpus
        self.dist = Normal(Variable(mean), Variable(std))  # noise variable
        self.lstm = nn.LSTM(input_size=corpus.embed_size,
                            hidden_size=self.input_encoding_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        self.output_linear = nn.Linear(self.input_encoding_size, corpus.vocab_size)
        self.features_linear = nn.Sequential(
            nn.Linear(cnn_output_size + len(mean), input_encoding_size),
            nn.ReLU()
        )

    def init_hidden(self, image_features):

        # generate rand
        rand = self.dist.sample_n(image_features.shape[0]).cuda()

        # hidden of shape (num_layers * num_directions, batch, hidden_size)
        hidden = self.features_linear(torch.cat((image_features, rand), 1).unsqueeze(0))

        # cell of shape (num_layers * num_directions, batch, hidden_size)
        cell = Variable(torch.zeros(image_features.shape[0], self.input_encoding_size).unsqueeze(0))

        return hidden.cuda(), cell.cuda()

    def forward(self, features, captions):
        states = self.init_hidden(features)
        hiddens, _ = self.lstm(captions, states)
        outputs = self.output_linear(hiddens[0])
        return outputs

    def sample_single_with_embedding(self, image_features):
        result = torch.zeros(self.max_sentence_length, self.embed.embed_size)
        hidden = self.init_hidden(image_features)
        inputs = self.embed.word_embedding(self.embed.START_SYMBOL)
        for i in range(self.max_sentence_length):
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.output_linear(hidden)
            predicted = outputs.max(1)[1]
            inputs = self.embed.word_embedding_from_index(predicted)
            result[i, :] = inputs
        return result

    def sample_with_embedding(self, images_features):
        batch_size = images_features.size(0)
        result = torch.zeros(batch_size, self.max_sentence_length, self.embed.embed_size).cuda()
        hidden = self.init_hidden(images_features)
        inputs = self.embed.word_embeddings([self.embed.START_SYMBOL] * batch_size)
        for i in range(self.max_sentence_length):
            result[:, i, :] = inputs
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.output_linear(hidden)
            predicted = outputs.max(1)[1]
            predicted = predicted.view(-1)
            inputs = self.embed.word_embeddings_from_indices(predicted).cuda()
        return result

    def save(self):
        torch.save({"state_dict": self.state_dict()}, FilePathManager.resolve("models/generator.pth"))

    @staticmethod
    def load(corpus: Corpus):
        state_dict = torch.load(FilePathManager.resolve("models/generator.pth"))
        state_dict = remove_module(state_dict["state_dict"])
        generator = ConditionalGenerator(corpus)
        generator.load_state_dict(state_dict)
        return generator

import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset.corpus import Corpus
from file_path_manager import FilePathManager


class Evaluator(nn.Module):
    def __init__(self, corpus: Corpus, cnn_features_size: int = 4096, input_encoding_size: int = 512,
                 num_layers: int = 1):
        super().__init__()
        self.corpus = corpus
        self.input_encoding_size = input_encoding_size
        self.lstm = nn.LSTM(corpus.embed_size, input_encoding_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(cnn_features_size, input_encoding_size),
            nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size):
        return Variable(torch.randn(1, batch_size, self.input_encoding_size)).cuda(), Variable(
            torch.randn((1, batch_size, self.input_encoding_size))).cuda()

    def forward(self, image_features, embeddings):
        batch_size = image_features.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(embeddings, hidden)
        hidden = hidden[0]
        hidden = hidden.view(batch_size, -1)
        image_features = self.linear(image_features)
        image_features = image_features.view(batch_size, -1)
        sim = torch.bmm(image_features.view(batch_size, 1, -1), hidden.view(batch_size, -1, 1))
        sim = self.sigmoid(sim)
        sim = sim.view(-1)
        return sim

    def save(self):
        torch.save({"state_dict": self.state_dict()}, FilePathManager.resolve("models/evaluator.pth"))

    @staticmethod
    def load(corpus: Corpus):
        state_dict = torch.load(FilePathManager.resolve("models/evaluator.pth"))
        state_dict = state_dict["state_dict"]
        evaluator = Evaluator(corpus)
        evaluator.load_state_dict(state_dict)
        return evaluator

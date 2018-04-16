import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset.corpus import Corpus
from file_path_manager import FilePathManager


class Evaluator(nn.Module):
    def __init__(self, corpus: Corpus, cnn_features_size: int = 4096, num_layers: int = 1):
        super().__init__()
        self.corpus = corpus
        self.cnn_features_size = cnn_features_size
        self.lstm = nn.LSTM(corpus.embed_size, self.cnn_features_size, num_layers=num_layers, batch_first=True)

    def init_hidden(self, batch_size):
        return Variable(torch.randn(1, batch_size, self.cnn_features_size)).cuda(), Variable(
            torch.randn((1, batch_size, self.cnn_features_size))).cuda()

    def forward(self, image_features, embeddings):
        hidden = self.init_hidden(image_features.shape[0])
        out, hidden = self.lstm(embeddings, hidden)
        hidden = hidden[0]
        hidden = hidden.view(image_features.size(0), -1)  # hidden = (batch_size, cnn_features_size)
        image_features = torch.t(
            image_features.view(image_features.size(0), -1))  # image_features = (cnn_features_size, batch_size)
        sim = hidden @ image_features  # sim = (batch_size, batch_size)
        sim = torch.diag(sim)  # (batch_size, 1)
        sim = sim.view(-1, 1)
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

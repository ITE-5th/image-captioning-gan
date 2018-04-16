import os
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.coco_dataset import CocoDataset
from dataset.corpus import Corpus
from extractor.vgg_extractor import VggExtractor
from file_path_manager import FilePathManager


class Evaluator(nn.Module):
    def __init__(self, corpus: Corpus):
        super().__init__()
        self.corpus = corpus
        self.lstm = nn.LSTM(corpus.embed_size, 4096, num_layers=1, batch_first=True)
        self.linear = nn.Linear(1, 1)

    def init_hidden(self):
        return Variable(torch.randn(1, 1, 4096)), Variable(torch.randn((1, 1, 4096)))

    def forward(self, image_features, embeddings):
        hidden = self.init_hidden()
        out, hidden = self.lstm(embeddings, hidden)
        hidden = hidden.view(hidden.size(0), -1)
        image_features = torch.t(image_features.view(image_features.size(0), -1))
        sim = hidden @ image_features
        sim = torch.diag(sim)
        return self.linear(sim)


if __name__ == '__main__':
    if not os.path.exists(FilePathManager.resolve("models")):
        os.makedirs(FilePathManager.resolve("models"))
    extractor = VggExtractor(use_gpu=False)
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
    print("Corpus loaded")
    evaluator = Evaluator(corpus).cuda()
    dataset = CocoDataset(corpus, extractor)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(evaluator.parameters(), lr=0.0001, weight_decay=1e-5)
    epochs = 5
    labels = Variable(torch.ones(batch_size))
    print("Begin Training")
    for epoch in range(epochs):
        for i, (images, captions) in enumerate(dataloader, 0):
            images, captions = Variable(images).cuda(), Variable(captions).cuda()
            optimizer.zero_grad()
            outputs = evaluator(images, captions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            torch.save({"state_dict": evaluator.state_dict()}, FilePathManager.resolve("models/evaluator.pth"))

import os
from multiprocessing import cpu_count

import torch
import torch.nn as nn
from pyro.distributions import Normal
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.coco_dataset import CocoDataset
from dataset.corpus import Corpus
from file_path_manager import FilePathManager
from vgg_extractor import VggExtractor


class Generator(nn.Module):

    def __init__(self, corpus: Corpus, mean: torch.FloatTensor = torch.zeros(1024),
                 std: torch.FloatTensor = torch.ones(1024),
                 max_sentence_length: int = 20):
        super().__init__()
        self.max_sentence_length = max_sentence_length
        self.corpus = corpus
        self.dist = Normal(Variable(mean), Variable(std))
        self.lstm = nn.LSTM(corpus.embed_size, 4096, num_layers=1, batch_first=True)
        self.linear = nn.Linear(corpus.embed_size, corpus.vocab_size)

    def init_hidden(self, image_features):
        # concat?
        return self.dist.sample() + image_features, Variable(torch.zeros(1, 1, 4096))

    def forward(self, image_features, logits=True):
        return self.sample(image_features, logits)

    def sample(self, image_features, return_logits):
        sampled_indices = []
        logits = []
        hidden = self.init_hidden(image_features)
        inputs = self.corpus.word_embedding(self.corpus.START_SYMBOL)
        done = False
        while not done:
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(outputs.squeeze(1))
            logits.append(outputs)
            predicted = outputs.max(1)[1]
            sampled_indices.append(predicted)
            inputs = self.corpus.word_embedding(predicted)
            done = predicted == self.corpus.word_index(self.dict.END_SYMBOL)
        sampled_indices = torch.cat(sampled_indices, 0)
        logits = torch.cat(logits, 0)
        return sampled_indices.squeeze() if not return_logits else logits.squeeze()


if __name__ == '__main__':
    if not os.path.exists(FilePathManager.resolve("models")):
        os.makedirs(FilePathManager.resolve("models"))
    extractor = VggExtractor(use_gpu=True)
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
    print("Corpus loaded")
    dataset = CocoDataset(corpus, extractor, evaluator=False)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    generator = Generator(corpus).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = Adam(generator.parameters(), lr=0.0001, weight_decay=1e-5)
    epochs = 20
    print("Begin Training")
    for epoch in range(epochs):
        for i, (images, captions) in enumerate(dataloader, 0):
            print(f"Batch = {i + 1}")
            images, captions = Variable(images).cuda(), Variable(captions).cuda()
            optimizer.zero_grad()
            outputs = generator.forward(images, logits=True)
            loss = criterion(outputs, captions)
            loss.backward()
            optimizer.step()
            torch.save({"state_dict": generator.state_dict()}, FilePathManager.resolve("models/generator.pth"))
        print(f"Epoch = {epoch + 1}")

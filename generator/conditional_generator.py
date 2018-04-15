import os
from multiprocessing import cpu_count

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.coco_dataset import CocoDataset
from dataset.corpus import Corpus
from extractor.vgg_extractor import VggExtractor
from file_path_manager import FilePathManager


class ConditionalGenerator(nn.Module):

    def __init__(self,
                 corpus: Corpus,
                 mean: torch.FloatTensor = torch.zeros(1024),
                 std: torch.FloatTensor = torch.ones(1024),
                 cnn_output_size: int = 4096,
                 input_encoding_size: int = 512,
                 max_sentence_length: int = 16,
                 num_layers: int = 1,
                 dropout: float = 0.5):
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

        # generate noise
        noise = self.dist.sample().unsqueeze(0)

        # hidden of shape (num_layers * num_directions, batch, hidden_size)
        hidden = self.features_linear(torch.cat((image_features, noise), 1).unsqueeze(0))

        # cell of shape (num_layers * num_directions, batch, hidden_size)
        cell = Variable(torch.zeros(1, image_features.shape[0], self.input_encoding_size))
        return hidden, cell

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        packed = pack_padded_sequence(embeddings, Variable(torch.IntTensor([self.max_sentence_length])),
                                      batch_first=True)
        hiddens, _ = self.lstm(packed, self.init_hidden(features))
        outputs = self.output_linear(hiddens.squeeze(1))
        return outputs

    # def forward(self, image_features, logits=True):
    #     assert (image_features.shape[1] == self.cnn_output_size)
    #
    #     return self.sample2(image_features, logits)

    def sample(self, image_features, return_logits):
        sampled_indices = []
        logits = []
        hidden = self.init_hidden(image_features)
        inputs = self.embed.word_embedding(self.embed.START_SYMBOL)
        done = False
        while not done:
            outputs, hidden = self.lstm(inputs, hidden)
            outputs = self.output_linear(outputs.squeeze(1))
            logits.append(outputs)
            predicted = outputs.max(1)[1]
            sampled_indices.append(predicted)
            inputs = self.embed.word_embedding(predicted)
            done = predicted == self.embed.word_index(self.dict.END_SYMBOL)
        sampled_indices = torch.cat(sampled_indices, 0)
        logits = torch.cat(logits, 0)
        return sampled_indices.squeeze() if not return_logits else logits.squeeze()

    def sample2(self, image_features, return_logits):
        sampled_indices = []
        logits = []
        states = self.init_hidden(image_features)
        inputs = self.embed(self.embed.START_SYMBOL)
        for _ in range(self.max_sentence_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.output_linear(hiddens.squeeze(1))
            logits.append(outputs)
            predicted = outputs.max(1)[1]
            sampled_indices.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        sampled_indices = torch.cat(sampled_indices, 0)
        logits = torch.cat(logits, 0)
        return sampled_indices.squeeze() if not return_logits else logits.squeeze()


if __name__ == '__main__':
    if not os.path.exists(FilePathManager.resolve("models")):
        os.makedirs(FilePathManager.resolve("models"))
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
    print("Corpus loaded")
    dataset = CocoDataset(corpus, evaluator=False)
    batch_size = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    generator = ConditionalGenerator(corpus).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = Adam(generator.parameters(), lr=0.0001, weight_decay=1e-5)
    epochs = 20
    print("Begin Training")
    for epoch in range(epochs):
        for i, (images, captions) in enumerate(dataloader, 0):
            print(f"Batch = {i + 1}")
            images, captions = Variable(images).cuda(), Variable(captions).cuda()

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            optimizer.zero_grad()
            outputs = generator.forward(images, inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            torch.save({"state_dict": generator.state_dict()}, FilePathManager.resolve("models/generator.pth"))
        print(f"Epoch = {epoch + 1}")

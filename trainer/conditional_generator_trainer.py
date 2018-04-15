import os
from os import cpu_count

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.corpus import Corpus
from dataset.test_dataset import TestDataset
from extractor.vgg_extractor import VggExtractor
from file_path_manager import FilePathManager
from generator.conditional_generator import ConditionalGenerator

if not os.path.exists(FilePathManager.resolve("models")):
    os.makedirs(FilePathManager.resolve("models"))

extractor = VggExtractor(use_gpu=True, pretrained=True)
corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
print("Corpus loaded")

batch_size = 2
dataset = TestDataset(corpus, extractor=extractor, evaluator=False)
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
        optimizer.zero_grad()
        outputs = generator.forward(images, captions)
        loss = criterion(outputs, captions)
        loss.backward()
        optimizer.step()
        torch.save({"state_dict": generator.state_dict()}, FilePathManager.resolve("models/generator.pth"))
    print(f"Epoch = {epoch + 1}")

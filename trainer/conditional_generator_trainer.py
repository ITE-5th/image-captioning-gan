import os
import time
from multiprocessing import cpu_count

from pretrainedmodels import utils
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader

# torch.backends.cudnn.enabled = False
from dataset.coco_dataset import CocoDataset
from dataset.corpus import Corpus
from extractor.vgg_extractor import VggExtractor
from file_path_manager import FilePathManager
from generator.conditional_generator import ConditionalGenerator

if not os.path.exists(FilePathManager.resolve("models")):
    os.makedirs(FilePathManager.resolve("models"))
extractor = VggExtractor(use_gpu=True)
tf_img = utils.TransformImage(extractor.cnn)
corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
print("Corpus loaded")

batch_size = 96
dataset = CocoDataset(corpus, evaluator=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())

generator = ConditionalGenerator(corpus).cuda()
criterion = nn.CrossEntropyLoss(ignore_index=corpus.word_index(corpus.PAD)).cuda()
optimizer = Adam(generator.parameters(), lr=0.0001, weight_decay=1e-5)
nn.MSELoss
epochs = 20

start = time.time()
print(f"number of batches = {len(dataset) // batch_size}")
print("Begin Training")
for epoch in range(epochs):
    for i, (images, inputs, targets) in enumerate(dataloader, 0):
        images = Variable(images).cuda()
        input = Variable(inputs)[:, :-1, :].cuda()
        target = Variable(targets)[:, 1:].cuda()
        input = pack_padded_sequence(input, [17] * images.shape[0], True)
        optimizer.zero_grad()
        outputs = generator.forward(images, input)
        target = target.contiguous().view(-1)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        end = time.time()
        print(f"Batch Time {end - start}")
        start = end
    print(f"Epoch = {epoch + 1}")
    generator.save()

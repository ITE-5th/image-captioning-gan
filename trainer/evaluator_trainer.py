import os
import time
from multiprocessing import cpu_count

from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from dataset.coco_dataset import CocoDataset
from dataset.corpus import Corpus
from evaluator.evaluator import Evaluator
from evaluator.evaluator_loss import EvaluatorLoss
from file_path_manager import FilePathManager
from generator.conditional_generator import ConditionalGenerator

if __name__ == '__main__':
    if not os.path.exists(FilePathManager.resolve("models")):
        os.makedirs(FilePathManager.resolve("models"))
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
    evaluator = Evaluator(corpus).cuda()
    generator = ConditionalGenerator.load(corpus).cuda()
    dataset = CocoDataset(corpus, evaluator=True)
    batch_size = 96
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    criterion = EvaluatorLoss().cuda()
    optimizer = optim.Adam(evaluator.parameters(), lr=0.001, weight_decay=1e-5)
    epochs = 5
    print(f"number of batches = {len(dataset) // batch_size}")
    print("Begin Training")
    start = time.time()
    for epoch in range(epochs):
        for i, (images, captions, other_captions) in enumerate(dataloader, 0):
            print(f"Batch = {i + 1}")
            images, captions, other_captions = Variable(images).cuda(), Variable(captions).cuda(), Variable(
                other_captions).cuda()
            captions = pack_padded_sequence(captions, [18] * batch_size, True)
            other_captions = pack_padded_sequence(other_captions, [18] * batch_size, True)
            optimizer.zero_grad()
            generator_outputs = generator.sample_with_embedding(images)
            evaluator_outputs = evaluator(images, captions)
            generator_outputs = evaluator(images, generator_outputs)
            other_outputs = evaluator(images, other_captions)
            loss = criterion(evaluator_outputs, generator_outputs, other_outputs)
            loss.backward()
            optimizer.step()
            end = time.time()
            print(f"Batch Time {end - start}")
            start = end
        print(f"Epoch = {epoch + 1}")
        evaluator.save()

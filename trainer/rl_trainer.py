import time
from multiprocessing import cpu_count

import torch
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
from policy_gradient.rl_loss import RLLoss

if __name__ == '__main__':
    epochs = 200
    batch_size = 128
    monte_carlo_count = 16
    corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
    evaluator = Evaluator.load(corpus).cuda()
    generator = ConditionalGenerator.load(corpus).cuda()
    dataset = CocoDataset(corpus, evaluator=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    evaluator_criterion = EvaluatorLoss().cuda()
    generator_criterion = RLLoss.apply
    evaluator_optimizer = optim.Adam(evaluator.parameters(), lr=1e-4, weight_decay=1e-5)
    generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4, weight_decay=1e-5)
    print(f"number of batches = {len(dataset) // batch_size}")
    print("Begin Training")
    for epoch in range(epochs):
        start = time.time()
        for i, (images, captions, other_captions) in enumerate(dataloader, 0):
            print(f"Batch = {i + 1}")
            images, captions, other_captions = Variable(images).cuda(), Variable(captions).cuda(), Variable(
                other_captions).cuda()
            # generator
            generator.unfreeze()
            evaluator.freeze()
            generator_optimizer.zero_grad()
            grads, rewards = generator.reward_forward(images, evaluator, monte_carlo_count=monte_carlo_count)
            loss = generator_criterion((grads, rewards))
            loss.backward()
            generator_optimizer.step()
            # evaluator
            evaluator.unfreeze()
            generator.freeze()
            temp = images.shape[0]
            generator_outputs = generator.sample_with_embedding(images)
            captions = torch.cat([captions, generator_outputs, other_captions])
            captions = pack_padded_sequence(captions, [18] * temp * 3, True)
            evaluator_optimizer.zero_grad()
            outs = evaluator(images, captions)
            captions, generator_captions, other_captions = outs[:temp], outs[temp:2 * temp], outs[2 * temp:]
            loss = evaluator_criterion(outs)
            loss.backward()
            evaluator_optimizer.step()
            end = time.time()
            print(f"Batch Time {end - start}")
            start = end
        print(f"Epoch = {epoch + 1}")
        evaluator.save()

import time
from multiprocessing import cpu_count

from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from dataset.corpus import Corpus
from dataset.evaluator_coco_dataset import EvaluatorCocoDataset
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
    dataset = EvaluatorCocoDataset(corpus)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    evaluator_criterion = EvaluatorLoss().cuda()
    generator_criterion = RLLoss().cuda()
    evaluator_optimizer = optim.Adam(evaluator.parameters(), lr=1e-4, weight_decay=1e-5)
    generator_optimizer = optim.Adam(generator.parameters(), lr=1e-4, weight_decay=1e-5)
    print(f"number of batches = {len(dataset) // batch_size}")
    print("Begin Training")
    for epoch in range(epochs):
        # generator
        generator.unfreeze()
        evaluator.freeze()
        for i, (images, captions, other_captions) in enumerate(dataloader, 0):
            print(f"Batch = {i + 1}")
            images, captions, other_captions = Variable(images).cuda(), Variable(captions).cuda(), Variable(
                other_captions).cuda()
            temp = images.shape[0]
            rewards, props = generator.reward_forward(images, evaluator, monte_carlo_count=monte_carlo_count)
            generator_optimizer.zero_grad()
            loss = generator_criterion(rewards, props)
            loss.backward()
            generator_optimizer.step()
        # evaluator
        evaluator.unfreeze()
        generator.freeze()
        for i, (images, captions, other_captions) in enumerate(dataloader, 0):       
            captions = pack_padded_sequence(captions, [18] * temp, True)
            other_captions = pack_padded_sequence(other_captions, [18] * temp, True)
            generator_outputs = generator.sample_with_embedding(images)
            evaluator_outputs = evaluator(images, captions)
            generator_outputs = evaluator(images, generator_outputs)
            other_outputs = evaluator(images, other_captions)
            loss = evaluator_criterion(evaluator_outputs, generator_outputs, other_outputs)
            loss.backward()
            evaluator_optimizer.step()
        print(f"Epoch = {epoch + 1}")
        generator.save()
        evaluator.save()

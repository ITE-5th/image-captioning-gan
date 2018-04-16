import pickle
import random

import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.corpus import Corpus
from extractor.vgg_extractor import VggExtractor
from file_path_manager import FilePathManager


class CocoDataset(Dataset):

    def __init__(self, corpus: Corpus, evaluator: bool = True):
        self.corpus = corpus
        self.evaluator = evaluator
        self.captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                          annFile=FilePathManager.resolve(
                                              f"data/annotations/captions_train2017.json"),
                                          transform=transforms.ToTensor())
        with open(FilePathManager.resolve("data/embedded_images.pkl"), "rb") as f:
            self.images = pickle.load(f)
        self.length = len(self.images) * 5

    def __getitem__(self, index):
        temp = index // 5
        image = self.images[temp]
        image = image.view(-1)
        item = self.captions[temp]
        caption = item[1][index % 5]
        caption = self.corpus.embed_sentence(caption, one_hot=False)
        if self.evaluator:
            other_index = random.choice([k for k in range(self.length // 5) if k != temp])
            other_caption = self.captions[other_index]
            other_index = random.choice(range(4))
            other_caption = other_caption[1][other_index]
            other_caption = self.corpus.embed_sentence(other_caption, one_hot=False)
            return image, caption, other_caption
        else:
            one_hot = self.corpus.sentence_indices(caption)
            return image, caption, one_hot

    def __len__(self):
        return self.length


if __name__ == '__main__':
    captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                 annFile=FilePathManager.resolve(
                                     f"data/annotations/captions_train2017.json"),
                                 transform=transforms.ToTensor())
    print(f"number of images = {len(captions.coco.imgs)}")
    extractor = VggExtractor(use_gpu=True)
    images = []
    i = 1
    for image, _ in captions:
        print(f"caption = {i}")
        item = extractor.forward(image).cpu().data
        images.append(item)
        i += 1
    with open(FilePathManager.resolve("data/embedded_images.pkl"), "wb") as f:
        pickle.dump(images, f)

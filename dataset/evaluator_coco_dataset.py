import pickle
import random

import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.corpus import Corpus
from file_path_manager import FilePathManager


class EvaluatorCocoDataset(Dataset):

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
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
        s = set(range(self.length // 5))
        s.remove(temp)
        s = list(s)
        other_index = random.choice(s)
        other_caption = self.get_captions(other_index)
        other_index = random.choice(range(5))
        other_caption = other_caption[1][other_index]
        other_caption = self.corpus.embed_sentence(other_caption, one_hot=False)
        return image, caption, other_caption

    def get_captions(self, index):
        coco = self.captions.coco
        img_id = self.captions.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]
        return target

    def __len__(self):
        return self.length

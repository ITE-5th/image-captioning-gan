import pickle

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
        print("loading finished")

    def __getitem__(self, index):
        image = self.images[index // 5]
        item = self.captions[index // 5]
        caption = item[1][index % 5]
        caption = self.corpus.embed_sentence(caption, one_hot=not self.evaluator)
        return image, caption

    def __len__(self):
        return self.length


if __name__ == '__main__':
    captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                 annFile=FilePathManager.resolve(
                                     f"data/annotations/captions_train2017.json"),
                                 transform=transforms.ToTensor())
    print(f"number of images = {len(captions.coco.imgs)}")
    extractor = VggExtractor(use_gpu=True, pretrained=True)
    images = []
    i = 1
    for image, _ in captions:
        print(f"caption = {i}")
        item = extractor.forward(image).cpu().data
        images.append(item)
        i += 1
    with open(FilePathManager.resolve("data/embedded_images.pkl"), "wb") as f:
        pickle.dump(images, f)
    # corpus = Corpus.load(FilePathManager.resolve("data/corpus.pkl"))
    # one_hot = []
    # i = 1
    # for _, capts in captions:
    #     print(f"caption = {i}")
    #     for capt in capts:
    #         one_hot.append(corpus.embed_sentence(capt, one_hot=True))
    #     i += 1
    # with open(FilePathManager.resolve("data/one_hot_sentences.pkl"), "wb") as f:
    #     pickle.dump(one_hot, f)
    # i = 1
    # embedded_sentences = []
    # for _, capts in captions:
    #     print(f"caption = {i}")
    #     for capt in capts:
    #         embedded_sentences.append(corpus.embed_sentence(capt, one_hot=False))
    #     i += 1
    # with open(FilePathManager.resolve("data/embedded_sentences.pkl"), "wb") as f:
    #     pickle.dump(embedded_sentences, f)

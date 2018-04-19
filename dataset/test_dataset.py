import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from dataset.corpus import Corpus
from extractor.vgg_extractor import VggExtractor
from file_path_manager import FilePathManager


class TestDataset(Dataset):

    def __init__(self, corpus: Corpus, extractor: VggExtractor, evaluator: bool = True):
        self.corpus = corpus
        self.evaluator = evaluator
        self.extractor = extractor
        self.captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                          annFile=FilePathManager.resolve(
                                              f"data/annotations/captions_train2017.json"),
                                          transform=transforms.ToTensor())

    def __getitem__(self, index):
        image, caption = self.captions[index]

        item = self.extractor.extract(image).cpu().data
        caption = self.corpus.embed_sentence(caption[0], one_hot=not self.evaluator)

        return item, caption

    def __len__(self):
        return len(self.captions)

# if __name__ == '__main__':
# path = FilePathManager.resolve("test_data.data")
# captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
#                              annFile=FilePathManager.resolve(
#                                  f"data/annotations/captions_train2017.json"),
#                              transform=transforms.ToTensor())
# captions = [captions[0], captions[1]]
# with open(path, "wb") as f:
#     pickle.dump(captions, f)

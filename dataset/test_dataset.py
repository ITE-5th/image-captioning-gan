import pickle

import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from file_path_manager import FilePathManager


class TestDataset(Dataset):

    def __init__(self, data=None):
        if data is None:
            data = pickle.load(FilePathManager.resolve("test_data.data"))
        self.data = [(image, cap) for image, capts in data for cap in capts]
        self.length = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    path = FilePathManager.resolve("test_data.data")
    captions = dset.CocoCaptions(root=FilePathManager.resolve(f'data/train'),
                                 annFile=FilePathManager.resolve(
                                     f"data/annotations/captions_train2017.json"),
                                 transform=transforms.ToTensor())
    captions = [captions[0], captions[1]]
    with open(path, "wb") as f:
        pickle.dump(captions, f)

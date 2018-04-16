import cv2
import numpy as np
import pretrainedmodels.utils as utils
import torch.nn as nn
import torchvision.transforms as transforms
from dlt.util import cv2torch
from pretrainedmodels import vgg16
from torch.autograd import Variable

from extractor.base_extractor import BaseExtractor
from file_path_manager import FilePathManager


class VggExtractor(BaseExtractor):

    def __init__(self, use_gpu: bool = True):
        super().__init__(use_gpu)
        self.use_gpu = use_gpu
        self.cnn = vgg16()
        self.trans = utils.TransformImage(self.cnn)
        self.trans = transforms.Compose([transforms.ToPILImage(), self.trans])
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        if use_gpu:
            self.cnn = self.cnn.cuda()
        self.cnn.eval()
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        if isinstance(image, np.ndarray):
            image = cv2torch(image)
        image = image.float()
        image = self.trans(image)
        if len(image.size()) == 3:
            image = image.unsqueeze(0)
        if self.use_gpu:
            image = image.cuda()
        temp = self.cnn(Variable(image))
        return temp


if __name__ == '__main__':
    extractor = VggExtractor()
    image_path = FilePathManager.resolve("test_images/image_1.png")
    print(extractor.forward(image_path))

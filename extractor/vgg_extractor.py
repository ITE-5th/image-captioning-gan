import cv2
import numpy as np
import torch
import torch.nn as nn
from dlt.util import cv2torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import vgg16

from extractor.base_extractor import BaseExtractor
from file_path_manager import FilePathManager


class VggExtractor(BaseExtractor):

    def __init__(self, use_gpu: bool = True, pretrained: bool = True):
        super().__init__(use_gpu)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        scale = 360
        input_shape = 224
        self.trans = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(scale),
                transforms.CenterCrop(input_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        self.use_gpu = use_gpu
        self.cnn = vgg16(pretrained=pretrained)
        self.cnn.classifier = nn.Sequential(*(self.cnn.classifier[i] for i in range(6)))
        if use_gpu:
            self.cnn = self.cnn.cuda()
        self.cnn.eval()
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2torch(image)
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

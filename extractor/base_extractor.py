from abc import abstractmethod, ABCMeta


class BaseExtractor(metaclass=ABCMeta):
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu

    @abstractmethod
    def forward(self, image):
        raise NotImplementedError()

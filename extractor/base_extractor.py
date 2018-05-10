from abc import abstractmethod, ABCMeta


class BaseExtractor(metaclass=ABCMeta):
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu

    @abstractmethod
    def extract(self, image):
        raise NotImplementedError()

    def __call__(self, image):
        return self.forward(image)

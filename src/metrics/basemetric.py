from abc import *



class BaseMetric(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def collect(self, pred, true):
        raise NotImplementedError

    @abstractmethod
    def summarize(self):
        raise NotImplementedError

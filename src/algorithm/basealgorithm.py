from abc import *



class BaseOptimizer(metaclass=ABCMeta):
    """Federated optimization algorithm.
    """
    @abstractmethod
    def step(self, closure=None):
        raise NotImplementedError
     
    @abstractmethod
    def accumulate(self, **kwargs):
        raise NotImplementedError

      
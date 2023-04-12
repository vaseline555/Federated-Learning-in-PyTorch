from abc import *



class BaseClient(metaclass=ABCMeta):
    """Class for client object having its own (private) data and resources to train a model.
    """
    def __init__(self, **kwargs):
        self.__identifier = None
        self.__model = None

    @property
    def id(self):
        return self.__identifier

    @id.setter
    def id(self, identifier):
        self.__identifier = identifier

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def download(self):
        raise NotImplementedError

    @abstractmethod
    def upload(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

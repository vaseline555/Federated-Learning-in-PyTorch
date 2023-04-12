from abc import *


class BaseServer(metaclass=ABCMeta):
    """Centeral server orchestrating the whole process of federated learning.
    """
    def __init__(self, **kwargs):
        self._round = 0
        self._model = None
        self._clients = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
    
    @property
    def round(self):
        return self._round

    @round.setter
    def round(self, round):
        self._round = round

    @property
    def clients(self):
        return self._clients

    @clients.setter
    def clients(self, clients):
        self._clients = clients

    @abstractmethod
    def _init_model(self, model):
        raise NotImplementedError

    @abstractmethod
    def _create_clients(self, client_datasets):
        raise NotImplementedError

    @abstractmethod
    def _broadcast_models(self, indices):
        raise NotImplementedError

    @abstractmethod
    def _sample_clients(self):
        raise NotImplementedError

    @abstractmethod
    def _request(self, indices, eval=False):
        raise NotImplementedError
    
    @abstractmethod
    def _aggregate(self, indices, update_sizes):
        raise NotImplementedError

    @abstractmethod
    def _central_evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def _adjust_lr(self):
        raise NotImplementedError
        
    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        raise NotImplementedError

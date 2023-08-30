from .fedavgclient import FedavgClient



class FedyogiClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedyogiClient, self).__init__(**kwargs)
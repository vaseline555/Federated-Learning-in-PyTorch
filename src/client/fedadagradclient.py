from .fedavgclient import FedavgClient



class FedadagradClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedadagradClient, self).__init__(**kwargs)
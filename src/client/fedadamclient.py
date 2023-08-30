from .fedavgclient import FedavgClient



class FedadamClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedadamClient, self).__init__(**kwargs)
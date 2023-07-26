import copy

from .fedavgclient import FedavgClient
from src import MetricManager


class FedavgmClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedavgmClient, self).__init__(**kwargs)
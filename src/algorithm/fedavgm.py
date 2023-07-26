import torch

from .fedavg import FedavgOptimizer



class FedavgmOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FedavgmOptimizer, self).__init__(params=params, **kwargs)

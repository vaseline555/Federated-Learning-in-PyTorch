import torch

from .fedavg import FedavgOptimizer



class FedproxOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FedproxOptimizer, self).__init__(params=params, **kwargs)

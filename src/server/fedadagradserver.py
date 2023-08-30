import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FedadagradServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedadagradServer, self).__init__(**kwargs)
        self.opt_kwargs = dict(
            beta=self.args.beta1, 
            v0=self.args.tau**2,
            tau=self.args.tau,
            lr=self.args.server_lr
        )

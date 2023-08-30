import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FedadamServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedadamServer, self).__init__(**kwargs)
        self.opt_kwargs = dict(
            betas=(self.args.beta1, self.args.beta2), 
            v0=self.args.tau**2,
            tau=self.args.tau,
            lr=self.args.server_lr
        )

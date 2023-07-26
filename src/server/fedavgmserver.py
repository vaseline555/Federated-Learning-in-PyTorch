
import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FedavgmServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedavgmServer, self).__init__(**kwargs)

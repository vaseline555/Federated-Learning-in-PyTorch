
import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FedproxServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedproxServer, self).__init__(**kwargs)

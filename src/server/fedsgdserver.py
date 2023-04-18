
import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FedsgdServer(FedavgServer):
    """Centeral server orchestrating the whole process of federated learning.
    """
    def __init__(self, **kwargs):
        super(FedsgdServer, self).__init__(**kwargs)

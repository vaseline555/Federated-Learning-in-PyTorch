
import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FedsgdServer(FedavgServer):
    """Centeral server orchestrating the whole process of federated learning.
    """
    def __init__(self, **kwargs):
        super(FedsgdServer, self).__init__(**kwargs)
    
    def _aggregate(self, ids, updated_sizes):
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')

        # calculate mixing coefficients according to sample sizes
        coefficients = {identifier: coefficient / sum(updated_sizes.values()) for identifier, coefficient in updated_sizes.items()}
        
        # accumulate weights
        for identifier in ids:
            locally_updated_weights_iterator = self.clients[identifier].upload()
            self.server_optimizer.accumulate(coefficients[identifier], locally_updated_weights_iterator)
            self.clients[identifier].model.zero_grad()
        else:
            self.server_optimizer.step()
            self.lr_scheduler.step()
        logger.info(f'[{self.args.algorithm.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')
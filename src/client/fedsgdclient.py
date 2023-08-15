import torch

from .fedavgclient import FedavgClient
from src import MetricManager



class FedsgdClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedsgdClient, self).__init__(**kwargs)

    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            
            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            for param in self.model.parameters():
                param.grad = None
            loss.backward()
            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            mm.track(loss.item(), outputs, targets)
        else:
            mm.aggregate(len(self.training_set), 1)
        return mm.results
    
    def upload(self):
        self.model.to('cpu')
        return self.model.named_parameters()

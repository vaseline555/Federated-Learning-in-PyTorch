import copy
import torch

from .fedavgclient import FedavgClient
from src import MetricManager


class FedproxClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedproxClient, self).__init__(**kwargs)

    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)
        
        global_model = copy.deepcopy(self.model)
        for param in global_model.parameters(): 
            param.requires_grad = False

        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                
                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                prox = 0.
                for name, param in self.model.named_parameters():
                    prox += (param - global_model.get_parameter(name)).norm(2)
                loss += self.args.mu * (0.5 * prox)

                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                
                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)
        return mm.results
    
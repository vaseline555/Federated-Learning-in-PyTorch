import torch
import inspect

from .fedavgclient import FedavgClient



class FedsgdClient(FedavgClient):
    """Class for client object having its own (private) data and resources to train a model.
    """
    def __init__(self, **kwargs):
        super(FedsgdClient, self).__init__(**kwargs)

    def update(self):
        self.model.train()
        self.model.to(self.args.device)
        
        update_results = dict()
        losses, corrects = 0., 0.
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            for param in self.model.parameters():
                param.grad = None
            loss.backward()

            losses += len(outputs) * loss.item()
            corrects += (outputs.argmax(1) == targets).sum().item()
        else:
            total_loss, total_acc = losses / len(self.training_set), corrects / len(self.training_set)
            update_results[1] = {'loss': total_loss, 'metrics': {'accuracy': total_acc}}
        return update_results
    
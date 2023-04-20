import copy

from .fedavgclient import FedavgClient



class FedproxClient(FedavgClient):
    def __init__(self, **kwargs):
        super(FedproxClient, self).__init__(**kwargs)

    def update(self):
        self.model.train()
        self.model.to(self.args.device)
        
        global_model = copy.deepcopy(self.model)
        for param in global_model.parameters(): 
            param.requires_grad = False

        update_results = dict()
        losses, corrects = 0., 0.
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            prox = 0.
            for name, param in self.model.named_parameters():
                prox += (param - global_model.get_parameter(name)).norm(2)
            loss += prox.mul_(self.args.mu).mul_(0.5)

            for param in self.model.parameters():
                param.grad = None
            loss.backward()

            losses += len(outputs) * loss.item()
            corrects += (outputs.argmax(1) == targets).sum().item()
        else:
            total_loss, total_acc = losses / len(self.training_set), corrects / len(self.training_set)
            update_results[1] = {'loss': total_loss, 'metrics': {'accuracy': total_acc}}
        return update_results
    
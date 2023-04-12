import torch
import inspect

from .baseclient import BaseClient



class Client(BaseClient):
    """Class for client object having its own (private) data and resources to train a model.
    """
    def __init__(self, args, training_set, test_set):
        """Client object is initiated by the center server."""
        super().__init__()
        self.args = args
        self.training_set = training_set
        self.test_set = test_set
        
        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset, shuffle):
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=shuffle)

    def update(self):
        self.model.train()
        self.model.to(self.args.device)
        
        update_results = dict()
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        for e in range(self.args.E):
            losses, corrects = 0., 0.
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                outputs = self.model(inputs)
                loss = self.criterion()(outputs, targets)

                for param in self.model.parameters():
                    param.grad = None
                loss.backward()
                optimizer.step()

                losses += len(outputs) * loss.item()
                corrects += (outputs.argmax(1) == targets).sum().item()
            else:
                epoch_loss, epoch_acc = losses / len(self.training_set), corrects / len(self.training_set)
                update_results[e + 1] = {'loss': epoch_loss, 'metrics': {'accuracy': epoch_acc}}
        return update_results

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()
        self.model.to(self.args.device)

        losses, corrects = 0., 0.
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            losses += len(outputs) * loss.item()
            corrects += (outputs.argmax(1) == targets).sum().item()
        else:
            total_loss, total_acc = losses / len(self.test_set), corrects / len(self.test_set)
            eval_results = {'loss': total_loss, 'metrics': {'accuracy': total_acc}}
        return eval_results

    def download(self, model):
        self.model.load_state_dict(model.state_dict())    

    def upload(self):
        return self.model.state_dict()
    
    def __len__(self):
        return len(self.training_set), len(self.test_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'

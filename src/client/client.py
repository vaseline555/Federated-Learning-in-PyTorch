import torch
import inspect



class Client:
    """Class for client object having its own (private) data and resources to train a model.
    """
    def __init__(self, args, client_id, training_set, test_set):
        """Client object is initiated by the center server."""
        self.args = args
        self.identifier = client_id
        self.training_set = training_set
        self.test_set = test_set
        self.__model = None
        
        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]

        self.train_loader = self._create_dataloader(self.training_set)
        self.test_loader = self._create_dataloader(self.test_set)

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def __len__(self):
        return len(self.training_set)

    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset):
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.args.B, shuffle=not self.args.no_shuffle)

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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += len(outputs) * loss.item()
                corrects += (outputs.argmax(1) == targets).sum().item()
            else:
                epoch_loss, epoch_acc = losses / len(self.training_set), corrects / len(self.training_set)
                update_results[e + 1] = {'loss': epoch_loss, 'acc': epoch_acc}
        return update_results
          
    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()
        self.model.to(self.args.device)

        eval_results = dict()
        losses, corrects = 0., 0.
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            outputs = self.model(inputs)
            loss = self.criterion()(outputs, targets)

            losses += len(outputs) * loss.item()
            corrects += (outputs.argmax(1) == targets).sum().item()
        else:
            epoch_loss, epoch_acc = losses / len(self.test_set), corrects / len(self.test_set)
            eval_results = {'loss': epoch_loss, 'acc': epoch_acc}
        return eval_results

    def __repr__(self):
        return f'CLIENT < {self.identifier} >'

import torch

from .basealgorithm import BaseOptimizer



class FedavgOptimizer(BaseOptimizer, torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        self.lr = kwargs.get('lr')
        self.momentum = kwargs.get('momentum', 0.)
        defaults = dict(lr=self.lr, momentum=self.momentum)
        super(FedavgOptimizer, self).__init__(params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            for param in group['params']:
                if param.data is None:
                    continue
                delta = self.state[param].pop('accum')
                if momentum > 0.:
                    if 'momentum_buffer' not in self.state[param]:
                        buffer = self.state[param]['momentum_buffer'] = torch.zeros_like(p).detach()
                    else:
                        buffer = self.state[param]['momentum_buffer']
                        buffer.mul_(momentum).add_(delta) # \delta w + \beta v
                    delta = buffer
                # apply update
                param.data.sub_(delta)
        return loss

    def accumulate(self, mixing_coefficient, local_param_iterator):
        for group in self.param_groups:
            for server_param, local_param in zip(group['params'], local_param_iterator):
                if 'accum' not in self.state[server_param]:
                    self.state[server_param]['accum'] = server_param.detach().sub(local_param.detach().mul(mixing_coefficient))
                else:
                    self.state[server_param]['accum'].sub_(local_param.detach().mul(mixing_coefficient))

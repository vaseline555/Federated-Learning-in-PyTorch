import torch

from .fedavg import FedavgOptimizer



class FedsgdOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FedsgdOptimizer, self).__init__(params=params, **kwargs)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta = param.grad.data
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
                if server_param.grad is None:
                    server_param.grad = local_param.grad.mul(mixing_coefficient)
                else:
                    server_param.grad.add_(local_param.grad.mul(mixing_coefficient))

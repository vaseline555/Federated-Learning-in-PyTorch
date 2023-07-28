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
            beta = group['momentum']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta = param.grad.mul(group['lr'])
                if beta > 0.:
                    if 'momentum_buffer' not in self.state[param]:
                        self.state[param]['momentum_buffer'] = torch.zeros_like(param).detach()
                    self.state[param]['momentum_buffer'].mul_(beta).add_(delta.mul(1. - beta)) # \beta * v + (1 - \beta) * (lr * grad)
                    delta = self.state[param]['momentum_buffer']
                param.data.sub_(delta)
        return loss

    def accumulate(self, mixing_coefficient, local_param_iterator):
        for group in self.param_groups:
            for server_param, (_, local_param) in zip(group['params'], local_param_iterator):
                local_delta = local_param.grad.mul(mixing_coefficient)
                if server_param.grad is None:
                    server_param.grad = local_delta
                else:
                    server_param.grad.add_(local_delta)

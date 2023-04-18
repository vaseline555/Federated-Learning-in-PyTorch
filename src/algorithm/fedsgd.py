import torch

from .fedavg import FedavgOptimizer



class FedsgdOptimizer(FedavgOptimizer):
    def __init__(self, params, **kwargs):
        super(FedsgdOptimizer, self).__init__(params=params, **kwargs)

    def step(self, closure=None):
        assert 'accum' in self.state[next(iter(self.state))] is not None, 'should call `accumulate` first!'
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            for param in group['params']:
                if param.grad is None:
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
                param.data.add_(delta, alpha=-self.lr)
        return loss

    def accumulate(self, mixing_coefficient, local_param_iterator):
        for group in self.param_groups:
            for server_param, local_param in zip(group['params'], local_param_iterator):
                if 'accum' not in self.state[server_param]:
                    self.state[server_param]['accum'] = local_param.grad.mul(mixing_coefficient)
                else:
                    self.state[server_param]['accum'].add_(local_param.grad.mul(mixing_coefficient))

import torch

from .basealgorithm import BaseOptimizer



class FedyogiOptimizer(BaseOptimizer, torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        lr = kwargs.get('lr')
        v0 = kwargs.get('v0')
        tau = kwargs.get('tau')
        momentum = kwargs.get('betas')
        defaults = dict(lr=lr, momentum=momentum, v0=v0, tau=tau)
        BaseOptimizer.__init__(self); torch.optim.Optimizer.__init__(self, params=params, defaults=defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for idx, group in enumerate(self.param_groups): 
            (beta1, beta2) = group['momentum']
            tau = group['tau']
            lr = group['lr']
            v0 = group['v0']
            for param in group['params']:
                if param.grad is None:
                    continue
                # get (\Delta_t)
                delta = -param.grad.data
                
                if idx == 0: # idx == 0: parameters; optimize according to algorithm
                    # calculate m_t
                    if 'momentum_buffer1' not in self.state[param]:
                        self.state[param]['momentum_buffer1'] = torch.zeros_like(param).detach()
                    self.state[param]['momentum_buffer1'].mul_(beta1).add_(delta.mul(1. - beta1)) # \beta1 * m_t + (1 - \beta1) * \Delta_t
                    m_new = self.state[param]['momentum_buffer1']
                    
                    # calculate v_t
                    if 'momentum_buffer2' not in self.state[param]:
                        self.state[param]['momentum_buffer2'] = v0 - delta.pow(2).mul(1. - beta2).mul((v0 - delta).sign())
                    v_curr = self.state[param]['momentum_buffer2']
                    self.state[param]['momentum_buffer2'].sub_(delta.pow(2).mul(1. - beta2).mul(v_curr.sub(delta.pow(2)).sign())) # v_t - (1 - \beta2) * \Delta_t^2 * sgn(v_t - \Delta_t)
                    v_new = self.state[param]['momentum_buffer2']

                    # update parameters
                    param.data.add_((m_new.div(v_new.pow(0.5).add(tau))).mul(lr))
                elif idx == 1: # idx == 1: buffers; just averaging
                    param.data.add_(delta)
        return loss

    def accumulate(self, mixing_coefficient, local_layers_iterator, check_if=lambda name: 'num_batches_tracked' in name):
        for group in self.param_groups: 
            for server_param, (name, local_signals) in zip(group['params'], local_layers_iterator):
                if check_if(name):
                    server_param.data.zero_()
                    server_param.data.grad = torch.zeros_like(server_param)
                    continue
                local_delta = (server_param - local_signals).mul(mixing_coefficient).data.type(server_param.dtype)
                if server_param.grad is None: # NOTE: grad buffer is used to accumulate local updates!
                    server_param.grad = local_delta
                else:
                    server_param.grad.data.add_(local_delta)

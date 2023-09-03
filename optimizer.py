from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
        
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
    
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['first_moment'] = torch.zeros_like(p.data)
                    state['second_moment'] = torch.zeros_like(p.data)
                
                #Update step
                state['step'] += 1
    
                # Update first and second moments of the gradients
                state['first_moment'] = group['betas'][0]*state['first_moment'] + (1-group['betas'][0])*grad
                state['second_moment'] = group['betas'][1]*state['second_moment'] + (1-group['betas'][1])*grad**2
                
                bias_correction1 = 1 - group['betas'][0] ** state['step']
                bias_correction2 = 1 - group['betas'][1] ** state['step']  
    
                # Update parameters
                step_size = group['lr'] * (bias_correction2)**0.5 / bias_correction1
                update = state["first_moment"] / (torch.sqrt(state["second_moment"]) + group['eps'])
                p.data = p.data -step_size * update
    
                # Update again using weight decay
                if group['weight_decay'] != 0:
                    p.data = p.data * (1 - group['lr'] * group['weight_decay'])
        
        return loss

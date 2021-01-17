# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:09:49 2020

@author: s6532600
"""
import numpy as np

class LRScheduler(object):
    
    def __init__(self):
        
        pass
        
    def get_lr(self):
        for i in self.learning_rates:
            yield i     

class LRConst(LRScheduler):
    
    def __init__(self, n_steps, lr):
        super(LRConst, self).__init__()
        self.learning_rates = [lr]*n_steps    

class LRLinearDecrease(LRScheduler):
    
    def __init__(self, n_steps, lb=1e-4, ub=1e-2):
        super(LRLinearDecrease, self).__init__()
        self.lb = lb
        self.ub = ub
        self.n_steps = n_steps
        self.learning_rates = np.linspace(lb,ub,n_steps)[::-1]     
    
class LRRangeTest(LRScheduler):
    
    def __init__(self, n_steps, lb=0.00001, ub=0.1):
        super(LRRangeTest, self).__init__()
        self.lb = lb
        self.ub = ub
        self.n_steps = n_steps
        self.learning_rates = np.linspace(lb,ub,n_steps)                    
            
class CLR(LRScheduler):
    
    def __init__(self, n_steps, lb=0.00001, ub=0.1, reserve=0.1):
        super(CLR, self).__init__()
        self.lb = lb
        self.ub = ub
        self.n_steps = n_steps
        self.reserve = reserve
        first_half = np.linspace(lb,ub,int(n_steps*(1-reserve)/2))
        second_half = first_half[::-1]
        n_steps_left = n_steps-len(first_half)-len(second_half)
        last_piece = np.linspace(lb/10,lb,n_steps_left)
        self.learning_rates = list(first_half)+list(second_half)+list(last_piece)
        
       
# -*- coding: utf-8 -*-
"""
Python helper functions
@author: thomas
"""
import os
import logging
import numpy as np

def make_logger(log_dir,name,level):
    ''' Initialize a root logger '''
    logger = logging.getLogger(name)

    # set the debugging level
    numeric_level = getattr(logging, level.upper(), None)
    logger.setLevel(numeric_level)

    # file handler
    fh = logging.FileHandler(os.path.join(log_dir,'logger.txt'),mode='w')
    # console handler
    ch = logging.StreamHandler()

    # formatter
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)    
    return logger

class TimedResults():
    ''' Collects statistics '''
    
    def __init__(self,n):
        self.t = []
        self.n = n
        self.data = []
        for i in range(n):
            self.data.append([])
    
    def add(self,t,*args):
        self.t.extend(t)
        for i in range(self.n):
            self.data[i].extend(args[i])
            
    def extract(self):
        results = [np.asarray(result) for result in self.data]
        output = [np.asarray(self.t)] 
        output.extend(results)
        return output
   
class Interval_checker:
    ''' Wrapper to check whether to update some quantity based on 
    t (timesteps) or ep (episodes), based on a specify total number of storage/copy moments '''
    def __init__(self,max_ep,t_max,store_copy_freq):
        self.ep_interval = np.ceil(max_ep/store_copy_freq)
        self.t_interval = np.ceil(t_max/store_copy_freq)
        self.ep_count = 0
        self.t_count = 0
        
    def should_update(self,t,ep):
        ''' check whether to update '''
        max_count = np.max([self.ep_count,self.t_count])
        if (ep // self.ep_interval) > self.ep_count:
            self.ep_count += 1
        if (t // self.t_interval) > self.t_count:
            self.t_count += 1
        if np.max([self.ep_count,self.t_count]) > max_count:
            return True
        else:
            return False   

class EMA:
    ''' Exponential moving average '''
    def __init__(self,mean=0,sd=1,rate=0.999):
        self.mean = mean
        self.sd = sd
        self.rate = rate
    
    def update(self,new_mean,new_sd):
        self.mean = self.rate*self.mean + (1-self.rate) * new_mean
        self.sd = self.rate*self.sd + (1-self.rate) * new_sd
    
    def extract(self):
        return self.mean,max(self.sd,1) 
        
class AnnealLinear():
    ''' Linear anneals between e_init and e_final '''

    def __init__(self,e_init,e_final,n):
        self.e_init = e_init
        self.e_final = e_final
        self.n = n
        
    def get(self,t):
        if t > self.n:
            return self.e_final
        else:
            return self.e_init + ( (t/self.n) * (self.e_final - self.e_init) )


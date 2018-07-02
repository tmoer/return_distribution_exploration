# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:33:17 2017

@author: thomas
"""
import numpy as np

class PartialRollout():
    ''' Collects roll out data and statistics '''
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.t = 0
        self.r_sum = 0
        self.terminal = False
        self.seed = None
    
    def add(self,state,action,reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.t += 1
        self.r_sum += reward[0]
    
    def add_last_state(self,state,terminal):
        self.last_state = state
        self.terminal = terminal
        
    def extract(self):
        return np.asarray(self.states),np.asarray(self.actions),np.asarray(self.rewards),np.asarray(self.last_state)  


# -*- coding: utf-8 -*-
"""
Dataset storage & batching
@author: Thomas Moerland, Delft University of Technology, The Netherlands
"""
import random
import numpy as np
import heapq
from collections import deque
import itertools

class Replay():
    ''' Replay databse. 
    Potential for prioritized replay by specifying a prioritization fraction >0.0 '''
    
    def __init__(self,max_size,prioritized_frac=0.0):
        self.max_size = max_size
        self.prioritized = prioritized_frac > 0.0
        self.prioritized_frac = prioritized_frac
        self.q = deque()
        self.size = 0
        if self.prioritized:
            self.q_p = []
        self.count = itertools.count() # add a unique counter to break ties

    def store_from_array(self,priority,*args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            if self.prioritized:
                self.put(entry,priority[i])
            else:
                self.put(entry,None)
        return self

    def store_from_array_prioritized(self,priority,*args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            if self.prioritized:
                self.put_prioritized(entry,priority[i])
        return self
        
    def put(self,tup,priority=None):
        # put normal queue
        if len(self.q) > self.max_size: # need to pop first
            self.q.popleft()
        self.q.append(tup)
        self.size += 1
        # put prior queue
        if self.prioritized:
            if len(self.q_p) > self.max_size: # need to pop first
                self.q_p = sorted(self.q_p)[:-1]
            heapq.heappush(self.q_p,(priority,next(self.count),tup))

    def put_prioritized(self,tup,priority=None):
        # put prior queue
        if self.prioritized:
            if len(self.q_p) > self.max_size: # need to pop first
                self.q_p = sorted(self.q_p)[:-1]
            heapq.heappush(self.q_p,(priority,next(self.count),tup))

    def get(self,n):
        minibatch = [heapq.heappop(self.q_p)[2] if (self.prioritized and (random.random() < self.prioritized_frac)) else random.sample(list(self.q),1)[0] for i in range(n)]
        return minibatch

    def sample_random_batch(self,n=32,return_arrays=True):
        if n <= self.size:
            batch = self.get(n)
            if return_arrays:
                arrays = []
                for i in range(len(batch[0])):
                    to_add = np.array([entry[i] for entry in batch])
                    arrays.append(to_add) 
                return tuple(arrays)
            else:
                return batch
        else:
            raise ValueError('Requested {} samples, but database only of size {}'.format(n,self.size))


class Database():
    ''' Database with iterator to generate minibatches. '''
    
    def __init__(self,data_size,batch_size,entry_type='sequential'):
        self.max_size = data_size        
        self.size = 0
        self.batch_size = batch_size
        self.insert_index = 0
        self.sample_index = 0
        self.experience = []
        self.entry_type = entry_type
    
    def get_insert_index(self):
        if self.entry_type == 'sequential':
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0
        elif self.entry_type == 'random':
            self.insert_index = random.randint(0,self.size-1)
        elif self.entry_type == 'prioritized':
            raise(NotImplementedError)
        return self.insert_index
    
    def clear(self):
        self.experience = []
        self.insert_index = 0
        self.sample_index = 0
        self.size = 0
    
    def store(self,experience):
        if self.size < self.max_size:
            self.experience.append(experience)
            self.size +=1
        else:
            self.experience[self.get_insert_index()] = experience

    def store_from_array(self,*args):
        for i in range(args[0].shape[0]):
            entry = []
            for arg in args:
                entry.append(arg[i])
            self.store(entry)
        return self
        
    def reset(self):
        self.sample_index = 0
        
    def shuffle(self):
        random.shuffle(self.experience)

    def sample_random_batch(self,n=32,return_arrays=True):
        if n <= self.size:
            batch = random.sample(self.experience,n)
            if return_arrays:
                arrays = []
                for i in range(len(batch[0])):
                    to_add = np.array([entry[i] for entry in batch])
                    arrays.append(to_add) 
                return tuple(arrays)
            else:
                return batch
        else:
            raise ValueError('Requested {} samples, but database only of size {}'.format(n,self.size))
            
    def __iter__(self):
        return self

    def __next__(self,batch_size=None,return_arrays=True):
        if batch_size is None: batch_size = self.batch_size
        
        if self.sample_index == 0:
            self.shuffle()

        if (self.sample_index + batch_size > self.size) and (not self.sample_index == 0):
            self.reset() # Reset for the next epoch
            raise(StopIteration)
          
        if (self.sample_index + 2*batch_size > self.size):
            batch = self.experience[self.sample_index:]
        else:
            batch = self.experience[self.sample_index:self.sample_index+batch_size]
        self.sample_index += batch_size
        
        if return_arrays:
            arrays = []
            for i in range(len(batch[0])):
                to_add = np.array([entry[i] for entry in batch])
                arrays.append(to_add) 
            return tuple(arrays)
        else:
            return batch
            
    next = __next__
            


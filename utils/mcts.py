# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search library, especially for combination with Tensorflow models
Applicable to continuous action space and stochastic, continuous state-space based on progressive widening
@author: Thomas Moerland, Delft University of Technology, The Netherlands
"""
import numpy as np
import random
import copy

class State():
    ''' State object '''
    def __init__(self,index,r=0.0,terminal=False,parent_action=None,depth=0):
        ''' initialize a new state '''
        self.index = index
        self.r = r
        self.terminal = terminal
        self.parent_action = parent_action
        self.depth = depth
        self.child_actions = []
        self.visits = 0

    def add_child_action(self,action_space,tolerance):
        ''' Add new action (expand) '''
        current_actions = [child_action.index for child_action in self.child_actions]
        new_action_index = action_space.sample()
        action_not_already_there = np.all([np.sum(np.abs(np.asarray(new_action_index) - np.asarray(action_index)))>tolerance for action_index in current_actions])
        if action_not_already_there:
            new_child_action = Action(new_action_index,parent_state=self)
            self.child_actions.append(new_child_action)
            return new_child_action
        else:
            return self.select()
        
    def select(self,sd=2):
        ''' Select one of the child actions based on UCT rule '''
        scores = [np.mean(child_action.values) + sd * np.std(child_action.values)*np.sqrt(2*np.log(self.visits)/np.float(child_action.visits))  if len(child_action.values) > 1 else np.mean(child_action.values) + sd * np.sqrt(2*np.log(self.visits)/np.float(child_action.visits)) for child_action in self.child_actions]
        #scores = [child_action.sum_value/child_action.visits + sd * np.sqrt(2*np.log(self.visits)/np.float(child_action.visits)) for child_action in self.child_actions]
        winners = np.argwhere(scores == np.max(scores)).flatten()        
        winner = random.choice(winners)
        return self.child_actions[winner]
        
    def act(self,next_state=None,tolerance=0.01):
        ''' Act based on MCTS output, returns the next state node '''
        if self.terminal: 
            return None
        action = self.child_actions[self.best_action_index]
        if next_state is None:
            return action.move() # random move on
        else:
            distances = [np.linalg.norm(np.array(child_state.index)-np.array(next_state)) for child_state in action.child_states]
            if np.min(distances) < tolerance: # can only be one minimum because child states have to differ
                next_index = np.argmin(distances)
                return action.child_states[next_index]
            else:
                return None # True next state not in tree, throw the tree away
            
    def update(self):
        self.visits += 1

class Action():
    ''' Action object '''
    def __init__(self,index,parent_state):
        self.index = index
        self.sum_value = 0 
        self.values = []
        self.visits = 0
        self.parent_state = parent_state
        self.child_states = []
        
    def add_child_state(self,depth,transition,tolerance,sess,model,Env):
        ''' Add child state (expand) '''
        next_state, r, terminal = transition(self.parent_state.index,self.index,sess=sess,model=model,Env=Env)   
        current_states = [child_state.index for child_state in self.child_states]
        state_not_already_there = np.all([np.sum(np.abs(np.asarray(next_state) - np.asarray(state_index)))>tolerance for state_index in current_states])
        if (Env is not None) and state_not_already_there and len(self.child_states)>0:
            print('Warning: Stochastic domain?')
        if state_not_already_there:
            new_child_state = State(next_state,r,terminal,self,depth)        
            self.child_states.append(new_child_state)
            return new_child_state
        else:
            return self.move()
            
    def move(self):
        ''' Select next state '''
        return random.choice(self.child_states)
        
    def update(self,r_sum):
        self.sum_value += r_sum
        self.visits += 1
        self.values.append(r_sum)
        
def trt(s,a,sess,model,**kwargs):
    ''' Transition, reward, terminal from model '''
    s1,r,terminal = sess.run([model.y_sample,model.r_sample,model.term_sample],
                             feed_dict = {model.x:s[None,:],
                                          model.a:a[None,:],
                                          model.y:s[None,:],
                                          model.k:1, # only one sample
                                          model.is_training:False, # sample from prior
                                          model.temp:0.001
                                          })
    return s1[0,], r[0,], terminal[0,]

def gym_trt_old(s,a,Env,**kwargs):
    ''' Gym simulator as transition function '''
    Env.reset()
    Env.state = s
    _,r,terminal,info = Env.step(a)
    s1 = Env.state
    return s1, r, terminal
    
def gym_trt(s,a,Env,**kwargs):
    ''' Gym simulator as transition function '''
    s1,r,terminal,info = Env.step(a)
    return s1, r, terminal

def rollout(state_index,steps,gamma,default_policy,transition=trt,sess=None,model=None,Env=None,action_space=None):
    ''' MCTS roll-out after expanding '''
    r_sum = 0
    for step in range(steps):
        if default_policy == 'targeted':
            next_action = sess.run(model.pi_sample,feed_dict = {model.x:state_index})
        elif default_policy == 'random':
            next_action = action_space.sample()
        next_state,r,terminal = transition(state_index,next_action,sess=sess,model=model,Env=Env)
        r_sum += (gamma**step)*r
        state_index = next_state
        if terminal:
            break
    return r_sum

def MCTS(root_index=None,root=None,N_traces=50,max_depth=50,action_space=None,C=2,alpha=0.4,default_policy='random',decision='max',transition=trt,policy=rollout,gamma=1,tolerance=0.1,sess=None,model=None,Env=None):
    ''' Monte Carlo Tree Search function '''
    if root == None:
        root = State(root_index) # initialize the root node
    else:
        root_index = root.index # continue from previous MCTS
        root.parent_action = None
        root.depth=0
        
    a_max = action_space.n if hasattr(action_space,'n') else 100
    
    for i in range(N_traces):
        node = root
        sim_env = copy.deepcopy(Env) if Env is not None else None
        depth = 0
                
        # Select (tree policy)
        while ((np.ceil(C * node.visits**alpha) < len(node.child_actions)) or (len(node.child_actions)==a_max)) and (not node.terminal) and (node.depth < max_depth):
            action = node.select()
            depth += 1
            # Select next node
            if np.ceil(C * action.visits**alpha) < len(action.child_states):
                node = action.move()
                if sim_env is not None: sim_env.step(action.index)
            else:
                node = action.add_child_state(depth,transition=transition,tolerance=tolerance,sess=sess,model=model,Env=sim_env)

        # Expand
        if (not node.terminal) and (node.depth < max_depth):
            action = node.add_child_action(action_space,tolerance=tolerance)
            depth += 1
            node = action.add_child_state(depth,transition=transition,tolerance=tolerance,sess=sess,model=model,Env=sim_env)
                    
        # Roll out with default policy
        if (not node.terminal) and (node.depth < max_depth):
            r_sum = policy(node.index,max_depth-node.depth,gamma=gamma,default_policy=default_policy,transition=transition,sess=sess,model=model,Env=sim_env,action_space=action_space)
        else:
            r_sum = 0.0
    
        # Back-up
        node.update()
        while node.parent_action is not None:
            r_sum = node.r + gamma * r_sum
            action = node.parent_action
            action.update(r_sum)     
            node = action.parent_state
            node.update()        
            
    # Return the estimates at the root
    root.actions = [child_action.index for child_action in root.child_actions]
    root.action_estimates = [action.sum_value/action.visits for action in root.child_actions]
    root.action_visits = [child_action.visits for child_action in root.child_actions]  
    root.action_uc = [child_action.sum_value/child_action.visits + 2.0 * np.sqrt(2*np.log(root.visits)/np.float(child_action.visits)) for child_action in root.child_actions]
    if decision == 'max':
        winners = np.argwhere(root.action_estimates == np.max(root.action_estimates)).flatten()
    elif decision == 'robust':
        winners = np.argwhere(root.action_visits == np.max(root.action_visits)).flatten()
    elif decision == 'ucb':
        winners = np.argwhere(root.action_uc == np.max(root.action_uc)).flatten()
    root.best_action_index = random.choice(winners)
    root.best_action = root.child_actions[root.best_action_index].index
    
    return root
    
class MCTS():
    
    def __init__(self,root_index=None,root=None,N_traces=50,max_depth=50,action_space=None,C=2,beta=0.4,default_policy='random',decision='max',transition=trt,policy=rollout,gamma=1,tolerance=0.1,sess=None,model=None,Env=None):
    ''' Monte Carlo Tree Search function '''
    self.root_index = root_index
    self.N_traces = N_traces
    self.max_depth = max_depth
    self.action_space = action_space
    self.C = C
    self.beta = beta
    self.default_policy = default_policy # random, targeted
    self.decision = decision
    self.network_type = network_type
    self.gamma = gamma
    self.tolerance = tolerance
    self.sess = sess
    self.model = model
    
    def run(self,Env=None):
    if root == None:
        root = State(root_index) # initialize the root node
    else:
        root_index = root.index # continue from previous MCTS
        root.parent_action = None
        root.depth=0
        
    a_max = action_space.n if hasattr(action_space,'n') else 100
    
    for i in range(N_traces):
        node = root
        sim_env = copy.deepcopy(Env) if Env is not None else None
        depth = 0
                
        # Select (tree policy)
        while ((np.ceil(C * node.visits**alpha) < len(node.child_actions)) or (len(node.child_actions)==a_max)) and (not node.terminal) and (node.depth < max_depth):
            action = node.select()
            depth += 1
            # Select next node
            if np.ceil(C * action.visits**alpha) < len(action.child_states):
                node = action.move()
                if sim_env is not None: sim_env.step(action.index)
            else:
                node = action.add_child_state(depth,transition=transition,tolerance=tolerance,sess=sess,model=model,Env=sim_env)

        # Expand
        if (not node.terminal) and (node.depth < max_depth):
            action = node.add_child_action(action_space,tolerance=tolerance)
            depth += 1
            node = action.add_child_state(depth,transition=transition,tolerance=tolerance,sess=sess,model=model,Env=sim_env)
                    
        # Roll out with default policy
        if (not node.terminal) and (node.depth < max_depth):
            r_sum = policy(node.index,max_depth-node.depth,gamma=gamma,default_policy=default_policy,transition=transition,sess=sess,model=model,Env=sim_env,action_space=action_space)
        else:
            r_sum = 0.0
    
        # Back-up
        node.update()
        while node.parent_action is not None:
            r_sum = node.r + gamma * r_sum
            action = node.parent_action
            action.update(r_sum)     
            node = action.parent_state
            node.update()        
            
    # Return the estimates at the root
    root.actions = [child_action.index for child_action in root.child_actions]
    root.action_estimates = [action.sum_value/action.visits for action in root.child_actions]
    root.action_visits = [child_action.visits for child_action in root.child_actions]  
    root.action_uc = [child_action.sum_value/child_action.visits + 2.0 * np.sqrt(2*np.log(root.visits)/np.float(child_action.visits)) for child_action in root.child_actions]
    if decision == 'max':
        winners = np.argwhere(root.action_estimates == np.max(root.action_estimates)).flatten()
    elif decision == 'robust':
        winners = np.argwhere(root.action_visits == np.max(root.action_visits)).flatten()
    elif decision == 'ucb':
        winners = np.argwhere(root.action_uc == np.max(root.action_uc)).flatten()
    root.best_action_index = random.choice(winners)
    root.best_action = root.child_actions[root.best_action_index].index
    
    return root
# -*- coding: utf-8 -*-
"""
Various policies
@author: thomas
"""
import numpy as np
import logging
import scipy.signal
from rl.policies import sample_net, get_net_mean, get_net_params, thompson_policy, egreedy_policy, ucb_policy, analytic_sd

logger = logging.getLogger('root')
logger.propagate = False

##### Bootstrapping #######

def bootstrap_terminal(hps,model):
    ''' returns bootstrap value/distribution parameters for a terminal state '''
    if hps.loss == 'sample':
        y_last = [[0.0]]    
        #if hps.output == 'categorical':
        #    y_last = np.ceil([[hps.n_bins/2]])        
    elif hps.loss == 'analytic':
        if hps.output == 'deterministic':
            y_last = [[0.0]]
        elif hps.output == 'gaussian':
            y_last = [[0.0,0.0001]] # mu and sigma
        elif hps.output == 'categorical':
            if True:
                y_last = np.clip(1.0 - (np.abs(model.transformer.means) / model.transformer.bin_width),0,1)[None,:]
            else:
                y_last = [0.0]*hps.n_bins                        
                index = model.transformer.to_index(0.0)[0] # find the bin that covers 0
                y_last[index] = 1.0 
                y_last = [y_last]
        elif hps.output == 'mog':
            y_last = [[1.0] + [0.0]*(hps.n_mix-1) + [0.0]*hps.n_mix + [0.0001]*hps.n_mix] # equal mixtures with mu and sigma at 0
    return np.array(y_last)    

def bootstrap_onpolicy(sess,model,sb,ab,seed,hps):
    ''' returns onpolicy bootstrap value/distributions parameters for non-terminal states'''

    if hps.loss == 'sample':
        y = sample_net(sess,model,sb,ab,seed,hps.p_dropout,hps.output)
        y_mean = np.mean(y)
    elif hps.loss == 'analytic':
        y = get_net_params(sess,model,sb,ab,seed,hps.p_dropout)
        y_mean = np.mean(get_net_mean(sess,model,sb,ab,seed,hps.p_dropout,hps.output))
    return y, y_mean

def bootstrap_offpolicy(sess,model,sb,action_dim,seed,hps,argmax_model,off_policy_on_mean):
    ''' returns offpolicy bootstrap value/distributions parameters for non-terminal states'''
    if argmax_model is None:
        argmax_model = model # No double DQN possible
    ab = offpolicy_argmax(sess,argmax_model,sb,action_dim,seed,hps,off_policy_on_mean)
    if hps.loss == 'sample':
        y = sample_net(sess,model,sb,ab,seed,hps.p_dropout,hps.output)
        if hps.output == 'categorical':
            # transform  
            y = model.transformer.to_value(y)
        y_mean = np.mean(y)
    elif hps.loss == 'analytic':
        y = get_net_params(sess,model,sb,ab,seed,hps.p_dropout)
        y_mean = np.mean(get_net_mean(sess,model,sb,ab,seed,hps.p_dropout,hps.output))
    return y,y_mean    

def offpolicy_argmax(sess,model,sb,action_dim,seed,hps,off_policy_on_mean):
    ''' returns the action index for the off policy decision 
    important parameter: if off_policy_on_mean = True, then we consider the mean of the output'''
        
    if hps.policy == 'thompson':
        a = thompson_policy(sb,model,sess,hps,seed,eval_on_mean_output=off_policy_on_mean,eval_on_mean_params=False) 
    elif hps.policy == 'egreedy':
        e = 0.0
        a = egreedy_policy(sb,model,sess,hps,e=e,seed=seed)
    elif hps.policy == 'ucb':
        a = ucb_policy(sb,model,sess,hps,seed,eval_on_mean_output=off_policy_on_mean,eval_on_mean_params=False) 
    return a

def bootstrap(sess,model,sb,ab,rb,last_state,terminal,hps,seed,argmax_model=None,off_policy=True,off_policy_on_mean=False):
    ''' Sample the network for the correct bootstrap targets
    returns bootstrap predictions y, may be either sample values, 
    or distributional targets '''
    action_dim = model.action_dim
    #print(sb.shape,ab.shape,rb.shape)
    if off_policy:
        # off policy
        if terminal:
            y,y_mean = bootstrap_offpolicy(sess,model,sb,action_dim,seed,hps,argmax_model,hps.off_policy_on_mean)
            y_last = bootstrap_terminal(hps,model)
            y = np.append(y,y_last,axis=0)
        else:
            state_rep = np.concatenate([sb,last_state[None,:]],axis=0)
            y,y_mean = bootstrap_offpolicy(sess,model,state_rep,action_dim,seed,hps,argmax_model,hps.off_policy_on_mean)    
    else:
        # on policy
        y,y_mean = bootstrap_onpolicy(sess,model,sb,ab,seed,hps)
        if terminal:
            y_last = bootstrap_terminal(hps,model)
        else:
            y_last,_ = bootstrap_offpolicy(sess,model,last_state[None,:],action_dim,seed,hps,argmax_model,hps.off_policy_on_mean)
        y = np.append(y,y_last,axis=0)
    return y,y_mean

def bootstrap_1step(sess,model,s1b,tb,hps,seed,argmax_model=None,off_policy_on_mean=False):
    ''' Off-policy 1-step predictions '''
    action_dim = model.action_dim
    #print(sb.shape,ab.shape,rb.shape)
    y,y_mean = bootstrap_offpolicy(sess,model,s1b,action_dim,seed,hps,argmax_model,hps.off_policy_on_mean)
    for i,t in enumerate(tb):
        if t:
            y[i] = bootstrap_terminal(hps,model)
    return y,y_mean

########## Propagation ############

def discount(x,gamma):
    ''' Generalized Advantage Estimation discounting '''
    return scipy.signal.lfilter([1],[1, -gamma],x[::-1],axis=0)[::-1]

def GAE(qsa,rb,gamma,lambda_):
    ''' GAE '''
    qsa = np.concatenate([[[0.0]],qsa])
    #print('qsa = {}'.format(qsa))
    TD = rb + gamma * qsa[1:] - qsa[:-1]
    y = discount(TD, gamma * lambda_) + qsa[:-1]
    return y
    
def propagate(sb,ab,rb,y_prime,hps,model,lambda_):
    ''' Propgate (distributional) targets in y through Bellman equation '''
    if hps.loss == 'sample':    
        # GAE
        y = GAE(y_prime,rb,hps.gamma,lambda_)
        if hps.output == 'categorical':
            y = model.transformer.to_index(y)[:,None]
    elif hps.loss == 'analytic':
        if hps.output == 'deterministic':
            # GAE
            y = GAE(y_prime,rb,hps.gamma,lambda_)
        elif hps.output == 'gaussian':
            # calculate (s,a),(mu,sigma) targets
            y = propagate_gaussian(rb,y_prime,hps)
        elif hps.output == 'categorical':    
            # calculate (s,a),(p1..pn) targets
            y = propagate_categorical2(rb,y_prime,model,hps)
        elif hps.output == 'mog':
            # calculate (s,a),(p1..pn,mu1..mun,s1...sn) targets
            y = propagate_mog(rb,y_prime,hps)
    #print(np.concatenate([sb,ab,rb,y],axis=1))
    return y
    
def propagate_gaussian(rb,y,hps):
    ''' Propagate Gaussian through Bellman equation '''
    y[:,0] = np.squeeze(rb,axis=1) + hps.gamma * y[:,0]
    y[:,1] = hps.gamma * y[:,1]
    return y

def softplus(x):
    return np.log(np.exp(x)+1)

def propagate_categorical2(rb,y,model,hps):
    ''' Propagate categorical through Bellman equation '''
    batch_size = rb.shape[0]
    # repeat means to match batch size
    means_expanded = np.repeat(model.transformer.means[None,:],batch_size,axis=0)

    # transform the means according to Bellman equation
    shifted_means = np.repeat(rb,model.transformer.n,axis=1) + hps.gamma * means_expanded
    shifted_means = np.clip(shifted_means,model.transformer.means[0],model.transformer.means[-1])

    # Repeat new means and bin means along third dimension
    shifted_means_rep = np.repeat(shifted_means[:,:,None],model.transformer.n,axis=2)
    bins_rep = np.repeat(means_expanded[:,None,:],model.transformer.n,axis=1)
    
    # project new means of the bin mean basis
    accountable = np.clip(1-(np.abs(shifted_means_rep - bins_rep)/model.transformer.bin_width),0,1)    

    # Redistribute the true probabilities according to accountability    
    prop_reps = np.repeat(y[:,:,None],model.transformer.n,axis=2)    
    new_density = np.sum(accountable * prop_reps,axis=1)
    return np.array(new_density)


def propagate_categorical(rb,y,model,hps):
    ''' Propagate categorical through Bellman equation '''
    out = []
    #print(rb,y,model.transformer.means)
    for i in range(rb.shape[0]): # loop over the minibatch
        new_means = rb[i][0] + hps.gamma * model.transformer.means # transfor means
        if True:
            new_means = np.clip(new_means,model.transformer.means[0],model.transformer.means[-1])
            new_means = np.repeat(new_means[:,None],model.transformer.n,axis=1)
            #print(new_means)            
            bins_rep = np.repeat(model.transformer.means[None,:],model.transformer.n,axis=0)
            #print(bins_rep)            
            accountable = np.clip(1-(np.abs(new_means - bins_rep)/model.transformer.bin_width),0,1)
            #print(accountable)            
            prop_reps = np.repeat(y[i,:][:,None],model.transformer.n,axis=1)
            #print(prop_reps)            
            new_density = np.sum(accountable * prop_reps,axis=0)
            #print(accountable,prop_reps,accountable*prop_reps,new_density)
            #print(new_density)
        else:            
            new_density,_ = np.histogram(new_means,bins=model.transformer.edges,weights=y[i,:])
        out.append(new_density)
    return np.array(out)

def propagate_mog(rb,y,hps):
    ''' Propagate mixture of Gaussians through Bellman equation 
    Each mixture propagates separately '''
    y[:,hps.n_mix:(2*hps.n_mix)] = np.repeat(rb,hps.n_mix,axis=1) + hps.gamma * y[:,hps.n_mix:(2*hps.n_mix)]
    y[:,(2*hps.n_mix):(3*hps.n_mix)] = hps.gamma * y[:,(2*hps.n_mix):(3*hps.n_mix)]
    return y
        
##### Wrapper for target calculation #####

def calculate_targets(data,D,D_sars,model,sess,hps,lambda_,off_policy,target_model):
    ''' process roll-out data to generate new training targets '''
    y_norm = []
    av_sds = []
    if hps.target_net:
        sample_model = target_model
        if hps.double_dqn:
            argmax_model = model # double dqn
        else:
            argmax_model = target_model
    else:
        sample_model = model
        argmax_model = None
    
    for rollout_data in data:
        sb,ab,rb,last_state = rollout_data.extract()  
                
        if hps.loss == 'sample':
            # Sample based propagation
            for l in range(hps.n_rep_mc): 
                seed = [np.random.randint(1e15),np.random.randint(1e15)] # set a new seed for MC integration
                for k in range(hps.n_rep_target): 
                    y_prime,y_mean = bootstrap(sess,sample_model,sb,ab,rb,last_state,rollout_data.terminal,hps,seed,argmax_model,off_policy,hps.off_policy_on_mean)
                    y = propagate(sb,ab,rb,y_prime[1:,],hps,model,lambda_)
                    D.store_from_array(sb,ab,y) # updated dataset 
                    y_norm.append(y_mean)
            # To add: Q-value normalization ?

        elif hps.loss == 'analytic':
            # Analytic propagation
            for l in range(hps.n_rep_mc): # MC integration over parametric uncertainty
                seed = [np.random.randint(1e15),np.random.randint(1e15)] # set a new seed for MC integration
                y_prime,y_mean = bootstrap(sess,sample_model,sb,ab,rb,last_state,rollout_data.terminal,hps,seed,argmax_model,off_policy,hps.off_policy_on_mean)
                y = propagate(sb,ab,rb,y_prime[1:,],hps,model,lambda_)
                D.store_from_array(sb,ab,y) # updated dataset 
                y_norm.append(y_mean)
        #print(np.concatenate((sb,ab,rb,y),axis=1))
        if hps.level == 'info':
            sds = analytic_sd(sess,model,sb,ab,seed,hps.p_dropout,hps.output)
            av_sds.append(np.mean(sds))
        
        # store to replay
        if hps.replay_size > 0:
            if hps.prioritized_frac > 0.0:
                # need to get the priorities based on the TD's
                seed = [np.random.randint(1e15),np.random.randint(1e15)] # just need to feed something new for each batch
                feed_dict = {model.x: sb,
                             model.a: ab,
                             model.y: y,
                             model.seed:seed}
                if hps.priority_type == 'loss':
                    prios = np.abs(sess.run(model.error,feed_dict = feed_dict))
                elif hps.priority_type == 'td':
                    net_mean = np.squeeze(get_net_mean(sess,model,sb,ab,seed,hps.p_dropout,hps.output),axis=1)
                    mean_y = mean_from_y(y,model,hps)
                    prios = np.abs(mean_y - net_mean)
                else:
                    raise KeyError('priority type {} unknown'.format(hps.priority_type))
            else:
                prios = None
            tt = np.zeros(sb.shape[0])
            tt[-1] = rollout_data.terminal
            sb1 = np.concatenate([sb[1:,],last_state[None,:]],axis=0)
            D_sars.store_from_array(prios,sb,ab,rb,sb1,tt)              

    av_sds = np.mean(av_sds)

    return D, D_sars, np.mean(y_norm), av_sds

def mean_from_y(y,model,hps):    
    if hps.output == 'gaussian':
        mean = y[:,0]
    elif hps.output == 'categorical':
        mean = np.matmul(y,model.transformer.means)
    elif hps.output == 'mog':
        mean = np.sum(y[:,:hps.n_mix]*y[:,hps.n_mix:(2*hps.n_mix)],axis=1)
    elif hps.output == 'deterministic':
        mean = np.squeeze(y,axis=1)
    return mean

def calculate_targets_off_policy(sb,ab,rb,s1b,tb,D,D_sars,model,sess,hps,target_model):
    ''' process roll-out data to generate new training targets '''
    y_norm = []
    if hps.target_net:
        sample_model = target_model
        if hps.double_dqn:
            argmax_model = model # double dqn
        else:
            argmax_model = target_model
    else:
        sample_model = model
        argmax_model = None
                    
    if hps.loss == 'sample':
        # Sample based propagation
        for l in range(hps.n_rep_mc): 
            seed = [np.random.randint(1e15),np.random.randint(1e15)] # set a new seed for MC integration
            for k in range(hps.n_rep_target): 
                y_prime,y_mean = bootstrap_1step(sess,sample_model,s1b,tb,hps,seed,argmax_model,hps.off_policy_on_mean)
                y = propagate(sb,ab,rb,y_prime,hps,model,lambda_=0.0)
                D.store_from_array(sb,ab,y) # updated dataset 
                y_norm.append(y_mean)
        # To add: Q-value normalization ?

    elif hps.loss == 'analytic':
        # Analytic propagation
        for l in range(hps.n_rep_mc): # MC integration over parametric uncertainty
            seed = [np.random.randint(1e15),np.random.randint(1e15)] # set a new seed for MC integration
            y_prime,y_mean = bootstrap_1step(sess,sample_model,s1b,tb,hps,seed,argmax_model,hps.off_policy_on_mean)
            y = propagate(sb,ab,rb,y_prime,hps,model,lambda_=0.0)
            D.store_from_array(sb,ab,y) # updated dataset 
            y_norm.append(y_mean)
    #print(np.concatenate((sb,ab,rb,y),axis=1))
            
    # store back to prioritized replay
    if hps.replay_size > 0:
        if hps.prioritized_frac > 0.0:
            # need to get the priorities based on the TD's
            seed = [np.random.randint(1e15),np.random.randint(1e15)] # just need to feed something new for each batch
            feed_dict = {model.x: sb,
                         model.a: ab,
                         model.y: y,
                         model.seed:seed}
            if hps.priority_type == 'loss':
                prios = np.abs(sess.run(model.error,feed_dict = feed_dict))
            elif hps.priority_type == 'td':
                net_mean = np.squeeze(get_net_mean(sess,model,sb,ab,seed,hps.p_dropout,hps.output))
                mean_y = mean_from_y(y,model,hps)
                prios = np.abs(mean_y - net_mean)
            else:
                raise KeyError('priority type {} unknown'.format(hps.priority_type))
            D_sars.store_from_array(prios,sb,ab,rb,s1b,tb)              
    return D,D_sars, np.mean(y_norm)



# -*- coding: utf-8 -*-
"""
Distribution functions
"""

import numpy as np
import tensorflow as tf
from network.graphtils import repeat
#from network.layers import fully_connected

def get_number_output_parameters(hps):
    ''' returns the number of necessary output distribution parameters '''
    if hps.output == 'deterministic':
        return 1
    elif hps.output == 'gaussian':
        return 2
    elif hps.output == 'categorical':
        return hps.n_bins
    elif hps.output == 'mog':
        return hps.n_mix * 3

def kl_divergence(mu1,sigma1,mu2,sigma2):
    ''' kl divergence for two univariate gaussians '''
    return tf.log(sigma2/sigma1)  + (tf.square(sigma1) + tf.square(mu1 - mu2))/(2*tf.square(sigma2)) - 0.5

def gaussian_ce(mu1,sigma1,mu2,sigma2):
    ''' crossentropy H(q(mu1,sigma1),p(mu2,sigma2)) '''
    return tf.log(2*np.pi*tf.square(sigma2))/2  + (tf.square(sigma1) + tf.square(mu1 - mu2))/(2*tf.square(sigma2))

def one_kl(mu1,sigma1,mu2,sigma2):
    return 0.5 * tf.log(np.pi*2 * tf.square(sigma2))  + (tf.square(sigma1) + tf.square(mu1 - mu2))/(2*tf.square(sigma2))

def bhattacharyya_distance(mu1,sigma1,mu2,sigma2):
    ''' bhattacharyya distance for two univariate gaussians '''
    return (tf.square(mu1-mu2)/(tf.square(sigma1) + tf.square(sigma2)))/4.0 + tf.log((tf.square(sigma1)/tf.square(sigma2) + tf.square(sigma2)/tf.square(sigma1) + 2.0)/4.0)/4.0

def hellinger_distance(mu1,sigma1,mu2,sigma2):
    ''' hellinger distance for two univariate gaussians '''
    return 1.0 - tf.sqrt(2*sigma1*sigma2/(tf.square(sigma1)+tf.square(sigma2))) * tf.exp(-0.25 * tf.square(mu1-mu2)/(tf.square(sigma1)+tf.square(sigma2)))

def output_distribution(z,hps,p_dropout,seed,kl,k):
    ''' Specifies losses and sampling operations for top graph layer '''
    if hps.output == 'deterministic':
        # params
        mu = z
        param = mu
        #dist
        mean = mu
        sample = mu
        # loss
        y = y_rep = tf.placeholder("float32", shape=[None,1],name='y')
        if hps.uncer == 'vi':
            y_rep = repeat(y,k)
        error = tf.reduce_sum(tf.square(mu - y_rep),axis=1)
        loss = tf.reduce_mean(error) + kl
        #loss = tf.losses.mean_squared_error(mu,y) + kl
        
    elif hps.output == 'gaussian':
        # params
        mu = z[:,0][:,None]
        log_sigma = z[:,1][:,None] + hps.sd_output_bias
        #sigma = tf.exp(log_sigma)
        sigma = tf.nn.softplus(log_sigma) 
        # dist
        outdist = tf.contrib.distributions.Normal(mu,sigma)
        sample = outdist.sample()
        mean = mu
        param = tf.concat([mu,sigma],axis=1)
        
        # loss            
        if hps.loss == 'analytic':
            y = y_rep = tf.placeholder("float32", shape=[None,2],name='y')
            if hps.uncer == 'vi':
                y_rep = repeat(y,k)            
            y_dist = tf.contrib.distributions.Normal(y_rep[:,0][:,None],y_rep[:,1][:,None])
            if hps.distance == 'kl':
                #kl_output = tf.contrib.distributions.kl_divergence(outdist,y_dist)
                kl_output = tf.contrib.distributions.kl_divergence(y_dist,outdist)
                #kl_output = kl_divergence(y[:,0],y[:,1],mu,sigma) # this should be the correct one? 
                #kl_output = kl_divergence(mu,sigma,y[:,0],y[:,1])
                #kl_output = one_kl(y[:,0],y[:,1],mu,sigma)
                #kl_output = one_kl(mu,sigma,y[:,0],y[:,1])
            elif hps.distance == 'bhat':
                kl_output = bhattacharyya_distance(mu,sigma,y_rep[:,0],y_rep[:,1])
            elif hps.distance == 'hel':
                kl_output = hellinger_distance(mu,sigma,y_rep[:,0],y_rep[:,1])
            elif hps.distance == 'ce':
                kl_output = y_dist.cross_entropy(outdist)
                #kl_output = gaussian_ce(mu,sigma,y_rep[:,0],y_rep[:,1])
            error = tf.reduce_sum(kl_output,axis=1)
            loss = tf.reduce_mean(error)  + kl
        elif hps.loss == 'sample':
            y = y_rep = tf.placeholder("float32", shape=[None,1],name='y')
            if hps.uncer == 'vi':
                y_rep = repeat(y,k)            
            error = -1.0 * outdist.log_prob(y_rep)
            loss = tf.reduce_mean(error) + kl
        
    elif hps.output == 'categorical':
        # params
        logits = z
        param = tf.nn.softmax(logits)
        # dist
        outdist = tf.contrib.distributions.Categorical(logits=logits)
        sample = tf.reshape(outdist.sample(1),[-1,1])
        mean = None

        # loss      
        if hps.loss == 'analytic':
            y = y_rep = tf.placeholder("float32", shape=[None,hps.n_bins],name='y')
            if hps.uncer == 'vi':
                y_rep = repeat(y,k)            
            #loss = -tf.reduce_sum(y * tf.log(param),axis=1)
            #loss = tf.reduce_mean(loss) + kl
            error = tf.nn.softmax_cross_entropy_with_logits(labels=y_rep,logits=logits)
            loss = tf.reduce_mean(error) + kl
        elif hps.loss == 'sample':
            y = y_rep = tf.placeholder("int32", shape=[None,1],name='y')
            if hps.uncer == 'vi':
                y_rep = repeat(y,k)  
            error = -1.0 * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_rep,logits=tf.expand_dims(logits,1))
            loss = tf.reduce_mean(error) + kl

    elif hps.output == 'mog':
        # params
        logits = z[:,:hps.n_mix]
        pi = tf.nn.softmax(logits)
        mu_p = z[:,hps.n_mix:(2*hps.n_mix)]
        bias = tf.tile(tf.constant(np.arange(hps.n_mix)-int(hps.n_mix/2),dtype='float32')[None,:],[tf.shape(mu_p)[0],1])
        mu_p = mu_p + bias            

        log_sigma = z[:,(2*hps.n_mix):(3*hps.n_mix)] + hps.sd_output_bias
        sigma_p = tf.nn.softplus(log_sigma) 
        param = tf.concat([pi,mu_p,sigma_p],axis=1)

        # dist
        p_dist = tf.contrib.distributions.Categorical(probs=pi)
        n_dist = []
        for i in range(hps.n_mix):
            n_dist.append(tf.contrib.distributions.Normal(mu_p[:,i],sigma_p[:,i]))  
        outdist = tf.contrib.distributions.Mixture(cat=p_dist,components=n_dist)
        sample = tf.reshape(outdist.sample(1),[-1,1])
        mean = outdist.mean()[:,None]

        # construct loss            
        if hps.loss == 'analytic':
            y = y_rep = tf.placeholder("float32", shape=[None,hps.n_mix*3],name='y')
            if hps.uncer == 'vi':
                y_rep = repeat(y,k)            
            qi,mu_q,sigma_q = tf.split(y_rep,3,axis=1)
            error = l2loss_gmm(pi,mu_p,sigma_p,qi,mu_q,sigma_q,hps.n_mix)
            loss = tf.reduce_mean(error) + kl
        elif hps.loss == 'sample':
            y = y_rep = tf.placeholder("float32", shape=[None,1],name='y')
            if hps.uncer == 'vi':
                y_rep = repeat(y,k)            
            error = -1.0 * outdist.log_prob(y_rep)
            loss = tf.reduce_mean(error) + kl
    
    return y,error,loss,sample,mean,param

def l2loss_gmm(pi,mu_p,sigma_p,qi,mu_q,sigma_q,n_mix):
    ''' Calculate L2 distance between two Gaussian mixture p(y) and q(y) '''
    piqi = tf.concat([pi,-1.0*qi],axis=1)
    s_pq = tf.concat([sigma_p,sigma_q],axis=1)
    mu_pq = tf.concat([mu_p,mu_q],axis=1)
    p_matrix = tf.einsum('ai,aj->aij',piqi,piqi) # outer product
    s_matrix = tf.tile(tf.expand_dims(s_pq,-1),[1,1,n_mix*2]) + tf.tile(tf.expand_dims(s_pq,1),[1,n_mix*2,1])
    distr = tf.contrib.distributions.Normal(loc = tf.tile(tf.expand_dims(mu_pq,1),[1,n_mix*2,1]), scale = s_matrix)
    pdfs = distr.prob(tf.tile(tf.expand_dims(mu_pq,-1),[1,1,n_mix*2]))
    return tf.reduce_sum(p_matrix * pdfs,axis=[1,2])

#def kl_gmm(pi,mu_p,sigma_p,qi,mu_q,sigma_q,n_mix):
#    ''' Sfikas et al, 2005 '''
#    V = 1.0/(1.0/sigma_p + 1.0/sigma_q)

class TransformDiscrete():
    ''' Transform categorical variable between integer values and true bins '''
    
    def __init__(self,n=51,min_val=-10,max_val=10):
        self.n = n
        self.min_val = min_val
        self.max_val = max_val
        self.edges = np.linspace(min_val,max_val,n+1)
        self.plot_edges = np.linspace(min_val,max_val,n+1)
        self.means = (self.edges[:-1] + self.edges[1:])/2
        self.edges[0] = -np.Inf
        self.edges[-1] = np.Inf
        self.bin_width = (max_val - min_val)/n

    def to_index(self,value):
        ''' from list of values to list of bin indices '''
        if type(value) == float:
            value = [value]
        return np.array([np.where(val>self.edges)[0][-1] for val in value])

    def to_value(self,indices):
        ''' from list of bin indices to list of values '''
        if type(indices) == int:
            indices = [indices]
        try:    
            return np.array([self.means[index] for index in indices])
        except:
            raise ValueError('bin index probably too large') 

def gaussian_diag_logps(mean, logvar, sample=None):
    if sample is None:
        noise = tf.random_normal(tf.shape(mean))
        sample = mean + tf.exp(0.5 * logvar) * noise
    return tf.clip_by_value(-0.5 * (np.log(2 * np.pi) + logvar + tf.square(sample - mean) / tf.exp(logvar)),-(10e10),10e10)
        
class DiagonalGaussian(object):
    def __init__(self, mean, logvar, sample=None):
        self.mean = mean
        self.logvar = logvar

        if sample is None:
            noise = tf.random_normal(tf.shape(mean))
            sample = mean + tf.exp(0.5 * logvar) * noise
        self.sample = sample

    def logps(self, sample):
        return gaussian_diag_logps(self.mean, self.logvar, sample)

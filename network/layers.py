# -*- coding: utf-8 -*-
"""
Custom nn layers
@author: thomas
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
#from network.normal import Normal
import numpy as np

def encoder(x,n_layers,n_hidden,n_final,batch_size,seed=[np.random.randint(1e15),np.random.randint(1e15)],activation_fn=tf.nn.elu,uncer='dropout',keep_prob=1.0,sigma_prior=1.0,kl_mode='analytic',prior_type='gaussian'):
    ''' Encoder. Implements n_layers, with n_hidden units in each of layers, and
    n_final * n_head units in the last layer '''
    kl_sum = 0.0
    if uncer == 'vi' or uncer == 'lin_bay':
        push_in = 'dropout' # only vi in the last layer
        keep_prob = 1.0
    else:
        push_in = uncer
        
    for i in range(n_layers-1):
        with tf.variable_scope('fc{}'.format(i)):
            x,kl = fully_connected(x,n_hidden,batch_size,seed,activation_fn,push_in,keep_prob,sigma_prior,kl_mode)
        kl_sum += kl
    # last layer
    #for n_hidden_last in [n_hidden,n_final]:
    if uncer == 'vi' or uncer == 'dropout':
        with tf.variable_scope('fc{}'.format(n_layers)):
            x,kl = fully_connected(x,n_final,batch_size,seed,activation_fn=None,uncer=uncer,keep_prob=1.0,sigma_prior=sigma_prior,kl_mode=kl_mode,prior_type=prior_type) # never dropout the outputs
        kl_sum += kl
        return x,kl_sum,None,None,None
    elif uncer == 'lin_bay':
        X_last = x
        with tf.variable_scope('fc{}'.format(n_layers)):
            x,W_mu,W_logsigma = fully_connected_linbay(x,n_final,seed) # never dropout the outputs
            kl_sum += 0.0
        return x,kl_sum,W_mu,W_logsigma,X_last
    
def fully_connected(x,n_hidden,batch_size,seed=[np.random.randint(1e15),np.random.randint(1e15)],activation_fn=tf.nn.elu,uncer='dropout',keep_prob=1.0,sigma_prior=1.0,kl_mode='analytic',prior_type='gaussian'):
    ''' Fully connected layer 
    x: input
    n_hidden: dimensionality of output
    uncer: 'dropout' or 'vi'
    keep_prob: dropout keep probability
    seed: seed for all random ops
    activation_fn: non-linearity to add
    sigma_prior: prior distribution for vi '''
    if uncer == 'dropout':
        x_out = fully_connected_dropout(x,n_hidden,seed,activation_fn,keep_prob)
        kl = 0.0
    elif uncer == 'vi':
        x_out, kl = fully_connected_variational(x,n_hidden,batch_size,seed,activation_fn,sigma_prior,kl_mode,prior_type)
    return x_out, kl
    
def fully_connected_dropout(x,n_hidden,seed=[np.random.randint(1e15),np.random.randint(1e15)],activation_fn=tf.nn.elu,keep_prob=1.0):
    ''' Fully connected layer with dropout '''
    enc = slim.fully_connected(x,n_hidden,activation_fn=activation_fn)
    x = slim.dropout(enc,keep_prob=keep_prob)#,seed=seed)
    return x

#############

def fully_connected_variational(x,n_hidden,batch_size,seed=[np.random.randint(1e15),np.random.randint(1e15)],activation_fn=tf.nn.elu,sigma_prior=1.0,kl_mode='analytic',prior_type='gaussian'):
    ''' Fully connected layer with variational inference on model parameters '''
    # get variables
    nin = x.get_shape()[1].value
    w_mu = tf.get_variable("w_mu", [nin, n_hidden], initializer=tf.contrib.layers.xavier_initializer())
    w_logsigma = tf.get_variable("w_sigma", [nin, n_hidden], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b_mu", [n_hidden], initializer=tf.contrib.layers.xavier_initializer())

    # variable distributions
    #if not prior_type == 'horseshoe':
    w_posterior = tf.distributions.Normal(w_mu,tf.nn.softplus(w_logsigma))
    #else:
    #    w_posterior = LogNormal_Posterior(w_mu,w_logsigma)

    # priors
    if prior_type == 'spike':
        w_prior = spike_slab_prior(x,n_hidden,sigma_prior)
    elif prior_type == 'gaussian':
        w_prior = normal_prior(x,n_hidden,sigma_prior)
    #elif prior_type == 'horseshoe':
    #    w_prior = horse_shoe_prior(x,n_hidden,sigma_prior)
    # w_prior,b_prior = horseshoe_prior(x,n_hidden,sigma_prior)
     
    # layer -- can't get tensor dependent seeding to work
    #tf.set_random_seed(seed)
    w = w_posterior.sample()#,seed=seed)
    #w=tf.Print(w,[w],message='These are the weights: ')
    
    z = tf.nn.bias_add(tf.matmul(x, w),b)
    
    # cannot feed batch_size in
#    w_mu = tf.get_variable("w_mu", [batch_size,nin, n_hidden], initializer=tf.contrib.layers.xavier_initializer())
#    w_logsigma = tf.get_variable("w_sigma", [batch_size,nin, n_hidden], initializer=tf.contrib.layers.xavier_initializer())
#    b = tf.get_variable("b_mu", [batch_size,n_hidden], initializer=tf.contrib.layers.xavier_initializer())
#
#    # variable distributions
#    w_posterior = tf.distributions.Normal(w_mu,tf.nn.softplus(w_logsigma))
#    w_prior = spike_slab_prior(x,n_hidden,sigma_prior)
#    w = w_posterior.sample()#,seed=seed)
#    z = tf.reduce_sum(x*w,axis=1) + b

    if activation_fn is not None:
        h = activation_fn(z)  
    else:
        h = z
    
    # kl term
    if kl_mode == 'analytic':
        kl_qp = tf.reduce_mean(tf.distributions.kl_divergence(w_posterior,w_prior)) # + tf.distributions.kl_divergence(b_dist,b_prior))
    elif kl_mode == 'sample':
        kl_qp = tf.reduce_mean(w_posterior.log_prob(w) - w_prior.log_prob(w)) # + b_dist.log_prob(b)  - b_prior.log_prob(b))
    return h, kl_qp

def normal_prior(x,n_hidden,sigma_prior):
    nin = x.get_shape()[1].value
    w_prior = tf.distributions.Normal(tf.constant(0.0,shape=[nin, n_hidden]),tf.constant(sigma_prior,shape=[nin, n_hidden]))
    return w_prior   

def spike_slab_prior(x,n_hidden,sigma_prior):
    nin = x.get_shape()[1].value
    mix = tf.constant(0.5,shape=[nin, n_hidden, 2])
    w_prior = tf.contrib.distributions.Mixture(
        cat=tf.distributions.Categorical(probs=mix),
        components=[
        tf.distributions.Normal(tf.constant(0.0,shape=[nin, n_hidden]),tf.constant(0.00001,shape=[nin, n_hidden])),
        tf.distributions.Normal(tf.constant(0.0,shape=[nin, n_hidden]),tf.constant(sigma_prior,shape=[nin, n_hidden]))
        ])
    return w_prior

def horse_shoe_prior(x,n_hidden,sigma_prior):
    nin = x.get_shape()[1].value
    sds = tf.constant(sigma_prior,shape=[nin, n_hidden])
    w_prior = HorseShoe2(sds)
    return w_prior

def kl_divergence_cauchy_lognormal(w,LogNormal_Posterior,HorseShoe_Prior):
    qw_z__pw_z = tf.distributions.kl_divergence(LogNormal_Posterior.Normal,HorseShoe_Prior.Normal)
    qz_pz = 0 # To implement
    return qw_z__pw_z + qz_pz
    
class HorseShoe:
    
    def __init__(self,scale):
        self.scale = scale
        self.HC = HalfCauchy(scale)
        sds1 = self.HC.sample(1)
        sds = sds1**2
        self.Normal = tf.distributions.Normal(loc=tf.zeros(tf.shape(sds),scale = sds))
    
    def sample(self):
        return self.Normal.sample(1)

class HalfCauchy:
    
    def __init__(self,scale):
        self.cauchy = tf.contrib.distributions.Cauchy(loc=tf.zeros(tf.shape(scale)),scale = scale)
    
    def sample(self):
        return tf.abs(self.cauchy.sample(1))
        
    def log_prob(self,x):
        return 2*self.cauchy.log_prob(x)

class HorseShoe2:
    
    def __init__(self,scale):
        self.scale = scale
        self.HC = HalfCauchy2(scale)
        sds = self.HC.sample()
        self.Normal = tf.distributions.Normal(loc=tf.zeros(tf.shape(sds)),scale = sds)
        self.sam = self.Normal.sample()
        self.log_p = self.Normal.log_prob(self.sam) + self.HC.log_prob()     
    
    def sample(self):
        return self.Normal.sample(),self.HC.sample()

    def log_prob(self,x):
        return self.log_p

class HalfCauchy2:
    
    def __init__(self,scale):
        self.gamma = tf.contrib.distributions.Gamma(0.5,scale**2)
        self.inv_gamma = tf.contrib.distributions.InverseGamma(0.5,1.0)
    
    def sample(self,z):
        return self.gamma.sample() * self.inv_gamma.sample()

    def log_prob(self,z1,z2):
        return self.gamma.log_prob(z1) + self.inv_gamma.log_prob(z2)
        
class LogNormal_Posterior:
    
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
        self.LN = LogNormal(mu,sigma)
        sds = self.LN.sample()**2
        self.Normal = tf.distributions.Normal(loc=tf.zeros(tf.shape(sds)),scale = sds)
        
    def sample(self):
        return self.Normal.sample(),self.LN.sample()

    def log_prob(self,x,z):
        return self.Normal.lob_prob(x),self.LN.log_prob(z)

class LogNormal:
    
    def __init__(self,mu,sigma):
        self.normal = tf.distributions.Normal(mu,sigma)
        
    def sample(self):
        return tf.exp(self.normal.sample())

    def log_prob(self,z):
        return self.normal.log_prob(z)

def horseshoe_prior(x,n_hidden,sigma_prior):
    nin = x.get_shape()[1].value
    # priors
    scale_global_a = tf.distributions.Gamma(0.5,sigma_prior)
    scale_global_b = tf.distributions.InverseGamma(0.5,1.0)
    
    w_scale_a = tf.distributions.Gamma(tf.constant(0.5,shape=[nin, n_hidden]),tf.constant(1.0,shape=[nin, n_hidden]))
    w_scale_b = tf.distributions.InverseGamma(tf.constant(0.5,shape=[nin, n_hidden]),tf.constant(1.0,shape=[nin, n_hidden]))
    
    w_prior = tf.distributions.Normal(tf.constant(0.0,shape=[nin, n_hidden]),tf.constant(1.0,shape=[nin, n_hidden]))

    w = w_prior.sample() + tf.sqrt(w_scale_a.sample() * w_scale_b.sample() * tf.constant(scale_global_a.sample(),shape=[nin, n_hidden]) * tf.constant(scale_global_b.sample(),shape=[nin, n_hidden]) )    
    return w

###############

def fully_connected_linbay(x,n_hidden,seed=[np.random.randint(1e15),np.random.randint(1e15)]):
    ''' fully connected layer for Bayesian inference '''
    nin = x.get_shape()[1].value
    W_mu = tf.get_variable("W_mu", [nin + 1, n_hidden], initializer=tf.contrib.layers.xavier_initializer())
    W_logsigma = tf.get_variable("W_logsigma", [nin + 1, n_hidden], initializer=tf.contrib.layers.xavier_initializer())

    # unpack
    w_mu = W_mu[:nin,:]
    b_mu = W_mu[nin,:]    
    w_logsigma = W_logsigma[:nin,:]
    b_logsigma = W_logsigma[nin,:]

    # variable distributions
    w_dist = tf.distributions.Normal(w_mu,tf.exp(w_logsigma))
    b_dist = tf.distributions.Normal(b_mu,tf.exp(b_logsigma))

    # layer
    w = w_dist.sample()#seed=seed)
    b = b_dist.sample()#seed=seed+13)
    z = tf.matmul(x, w) + b
    # no non-linearity

    return z,W_mu,W_logsigma
    
def make_linbay_update_op(W_mu,W_logsigma,X,Y,hps,n_out):
    ''' estimate and copy the new parameter uncertainties'''
    S,m = get_new_mean_S_for_w(X,Y,hps,n_out)
    assign_op = tf.group([tf.assign(W_mu,m),tf.assign(W_logsigma,tf.log(S))])
    return assign_op

def get_new_mean_S_for_w(X,Y,hps,n_out):
    ''' Bayesian linear regression '''
    sigma_y = 1.0 # fix this for now
    S = tf.matrix_inverse(tf.matmul(tf.transpose(X),X)/sigma_y + tf.diag(tf.tile(tf.constant([1/tf.square(hps.sigma_prior)]),n_out)))
    m = tf.matmul(tf.matmul(S,tf.transpose(X)),Y)/sigma_y
    return S,m 


def conv_encoder(x,hps):
    ''' Convolutional encoder '''
    conv1 = slim.conv2d(activation_fn=tf.nn.elu,
            inputs=x,num_outputs=32,
            kernel_size=8,stride=4,padding='VALID')
    conv2 = slim.conv2d(activation_fn=tf.nn.elu,
            inputs=conv1,num_outputs=64,
            kernel_size=4,stride=2,padding='VALID')
    conv3 = slim.conv2d(activation_fn=tf.nn.elu,
            inputs=conv2,num_outputs=32,
            kernel_size=3,stride=1,padding='VALID')
    return slim.flatten(conv3)

def weight_norm_dense(x, size, name, init_scale=1.0):
    ''' Weight normalized fully connected '''
    v = tf.get_variable(name + "/V", [int(x.get_shape()[1]), size],
                        initializer=tf.random_normal_initializer(0, 0.05))
    g = tf.get_variable(name + "/g", [size], initializer=tf.constant_initializer(init_scale))
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(0.0))

    # use weight normalization (Salimans & Kingma, 2016)
    x = tf.matmul(x, v)
    scaler = g / tf.sqrt(sum(tf.square(v), axis=0, keepdims=True))
    return tf.reshape(scaler, [1, size]) * x + tf.reshape(b, [1, size])    
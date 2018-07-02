# -*- coding: utf-8 -*-
"""
Network specification

@author: thomas
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from network.layers import  encoder, conv_encoder, make_linbay_update_op
from network.graphtils import make_copy_from_to, is_convolutional, get_nonlin, add_epsilon, add_optimizer, sync, repeat
from network.distributions import output_distribution, TransformDiscrete, get_number_output_parameters

def make_network(hps,cluster):
    # global model
    if hps.distributed:
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:{}/cpu:0".format(hps.job_index),cluster=cluster)):
            with tf.variable_scope('global'):
                global_model = Network(hps,'global_model')
                with tf.variable_scope('global_counters'):
                    global_t = tf.get_variable("global_t", [], tf.int64, initializer=tf.constant_initializer(0, dtype=tf.int64),trainable=False)
                    global_ep = tf.get_variable("global_ep", [], tf.int64, initializer=tf.constant_initializer(0, dtype=tf.int64),trainable=False)
                # global target net
                if hps.target_net:
                    global_target_model = Network(hps,'global_target_model')            
                    global_copy_op = make_copy_from_to(global_model.var_list,global_target_model.var_list)
                else:
                    global_target_model,global_copy_op = None,None
    else:
        with tf.variable_scope('global'):
            global_model,global_target_model,global_copy_op = None,None,None
            global_t = tf.get_variable("global_t", [], tf.int64, initializer=tf.constant_initializer(0, dtype=tf.int64),trainable=False)
            global_ep = tf.get_variable("global_ep", [], tf.int64, initializer=tf.constant_initializer(0, dtype=tf.int64),trainable=False)

    # local_model
    if hps.distributed:
        with tf.device("/job:worker/task:{}/cpu:0".format(hps.job_index)):
            with tf.variable_scope('local'):
                model = Network(hps,'local_model') # Import network
                if hps.target_net:
                    target_model = Network(hps,'local_target_model')            
                    copy_op = make_copy_from_to(model.var_list,target_model.var_list)
                else:
                    target_model,copy_op = None,None
                
                # Loss and optimizer    
                model,global_model = add_optimizer(model,hps,global_model)
                
        # global t for distributed training
        model.global_t = global_t
        model.t = t = tf.placeholder('int64')    
        model.inc_t = model.global_t.assign_add(t)
        model.global_ep = global_ep
        model.ep = ep = tf.placeholder('int64')    
        model.inc_ep = model.global_ep.assign_add(ep)
        
        add_epsilon(model,hps)
                                
        # copy between global and local
        sync_op = sync(model,target_model,global_model,global_target_model)
    else:
        model = Network(hps,'local_model') # Import network
        if hps.target_net:
            target_model = Network(hps,'local_target_model')            
            copy_op = make_copy_from_to(model.var_list,target_model.var_list)
        else:
            target_model,copy_op = None,None       

        # global t for distributed training
        model.global_t = global_t
        model.t = t = tf.placeholder('int64')    
        model.inc_t = model.global_t.assign_add(t)
        model.global_ep = global_ep
        model.ep = ep = tf.placeholder('int64')    
        model.inc_ep = model.global_ep.assign_add(ep)
        
        add_epsilon(model,hps)
        
        # Loss and optimizer    
        model,global_model = add_optimizer(model,hps,global_model)
        
        # copy between global and local
        sync_op = sync(model,target_model,global_model,global_target_model)
        
    return model, target_model, copy_op, global_model, global_target_model, global_copy_op, sync_op
 
class Network(object):
    ''' Parametric network specification ''' 
    
    def __init__(self,hps,scope):
        # Network
        with tf.variable_scope(scope):
            # Check state & action spaces
            self.action_dim, self.action_discrete  = hps.action_dim, hps.action_discrete
            self.state_dim, self.state_discrete  = hps.state_dim, hps.state_discrete
            
            # placeholders
            if not self.state_discrete:
                self.x = x = tf.placeholder("float32", shape=np.append(None,hps.state_dim),name='x') # s   
            else:
                self.x = x = tf.placeholder("int32", shape=np.append(None,1)) # s 
                x =  tf.squeeze(tf.one_hot(x,hps.state_dim,axis=1),axis=2)
            self.a = a = tf.placeholder("int32", shape=[None,1],name='a') # a
            a_one = tf.squeeze(tf.one_hot(a,hps.action_dim,axis=1),axis=2)
            self.seed = seed = tf.placeholder('int64',[2],name='seed')
            self.p_dropout = p_dropout = tf.Variable(hps.p_dropout,trainable=False,name='p_dropout')
            self.k = k = tf.Variable(1,trainable=False,name='k')
            self.batch_size = batch_size = tf.Variable(1,dtype='int64',trainable=False,name='batch_size')

            if hps.uncer == 'vi':
                x = repeat(x,k)
                a_one = repeat(a_one,k)
            
            # Representation layers for large domains (no dropout)
            convolutional = is_convolutional(hps)
            if convolutional:
                x = conv_encoder(x,hps)
            
            n_final = get_number_output_parameters(hps)
            # Fully connected layers with dropout
            if hps.uncer == 'lin_bay':
                print('Warning, setting network head to single because uncer == lin_bay')
                hps.network == 'single'
                
            if hps.network == 'single':
                xa_con = tf.concat([x,a_one],axis=1)
                xa_enc,kl_enc,W_mu,W_logsigma,X_last = encoder(xa_con,hps.n_layers,int(hps.n_hidden*(self.action_dim/2)),n_final=n_final,batch_size=batch_size,seed=seed,activation_fn=get_nonlin(hps),uncer=hps.uncer,keep_prob=p_dropout,sigma_prior=hps.sigma_prior,kl_mode=hps.kl_mode,prior_type=hps.prior_type)
                self.y,self.error,self.loss,self.sample,self.mean,self.params = output_distribution(xa_enc,hps,p_dropout,seed,kl_enc,k)
                
                # add lin_bayes op
                n_out = self.y.get_shape()[1].value
                self.lin_bay_update = make_linbay_update_op(W_mu,W_logsigma,X_last,self.y,hps,n_out)               
                
            elif hps.network == 'multiple':
                x = slim.flatten(x)
                x_encs = []
                kl = 0.0
                for i in range(hps.action_dim):
                    with tf.variable_scope('head{}'.format(i)):
                        x_enc,kl_enc,_,_,_ = encoder(x,hps.n_layers,hps.n_hidden,n_final=n_final,batch_size=batch_size,seed=seed+3*i,activation_fn=get_nonlin(hps),uncer=hps.uncer,keep_prob=p_dropout,sigma_prior=hps.sigma_prior,kl_mode=hps.kl_mode,prior_type=hps.prior_type)
                    x_encs.append(x_enc)
                    kl += kl_enc
                x_encs = [tf.expand_dims(x_enc,axis=2) for x_enc in x_encs]
                x_enc = tf.concat(x_encs,axis=2)
                a_tiled = tf.tile(tf.expand_dims(a_one,1),[1,n_final,1])
                xa_enc = tf.reduce_sum(x_enc * a_tiled,axis=2)
                self.y,self.error,self.loss,self.sample,self.mean,self.params = output_distribution(xa_enc,hps,p_dropout,seed+19*i,kl,k)
            
            if hps.output == 'categorical':
                self.transformer = TransformDiscrete(n=hps.n_bins,min_val=hps.cat_min,max_val=hps.cat_max)
                
            # var list
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
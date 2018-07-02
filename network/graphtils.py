# -*- coding: utf-8 -*-
"""
Partially from 
https://github.com/openai/iaf/blob/master/tf_utils/common.py
"""
import tensorflow as tf
import numpy as np

def clip_gradients(grads,value):
    return [tf.clip_by_value(grad, -1.0*value, value) for grad in grads]

def add_optimizer(model,hps,global_model=None):
    ''' Adds optimizer to the model, including gradient clipping and global_model copying '''
    # Learning rate
    if hps.decay_lr == 'None':
        lr = tf.Variable(hps.lr,name="learning_rate",trainable=False)
    elif hps.decay_lr == 'inverse':
        lr = tf.train.inverse_time_decay(hps.lr,model.global_ep,1,decay_rate=hps.decay_rate)
    elif hps.decay_lr == 'linear':        
        lr = tf.train.polynomial_decay(hps.lr, model.global_ep,
                                          hps.max_ep*0.7, 0.1*hps.lr,
                                          power=1.0)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)

    # Gradients
    grads = tf.gradients(model.loss, model.var_list)
    grad_norm = tf.global_norm(grads)
    if hps.clip_gradients:
        if hps.clip_global > 0.0:
            clip_global = tf.Variable(hps.clip_global,trainable=False)
            grads,model.gradient_norm = tf.clip_by_global_norm(grads, clip_global, use_norm=grad_norm)
            model.clipped_norm = tf.global_norm(grads)
        else:
            grads = clip_gradients(grads,hps.clip_value)
            ngrads = sum([1.0 for grad in grads])
            model.gradient_norm = grad_norm / ngrads
            model.clipped_norm = tf.global_norm(grads) / ngrads
    else:
        model.gradient_norm = model.clipped_norm = grad_norm
        
    # Train op    
    gvs = list(zip(grads, model.var_list)) if not hps.distributed else list(zip(grads, global_model.var_list))
    model.train_op = optimizer.apply_gradients(gvs)
    
    return model, global_model

def sync(model,target_model,global_model,global_target_model): 
    ''' create sync_op between global and local models '''
    if global_model is not None:
        oplist = [v1.assign(v2) for v1, v2 in zip(model.var_list, global_model.var_list)]
        if target_model is not None:
            oplist.extend([v1.assign(v2) for v1, v2 in zip(target_model.var_list, global_target_model.var_list)])
        sync = tf.group(*oplist)        
    else:
        sync = None # nothing to sync
    return sync

def get_nonlin(hps):
    ''' Select nonlinearity from string '''
    if hps.nonlin == 'elu':
        return tf.nn.elu
    elif hps.nonlin == 'relu':
        return tf.nn.relu
    elif hps.nonlin == 'tanh':
        return tf.nn.tanh

def add_epsilon(model,hps):
    ''' add epsilon decay schedule to the graph '''
    model.epsilon = tf.train.polynomial_decay(hps.e_init, model.global_ep,
                                          hps.max_ep*hps.anneal_frac, hps.e_final,
                                          power=2.0)

def is_convolutional(hps):
    ''' verify whether Env requires convolutions '''
    if type(hps.state_dim) ==  int:
        return False
    convolutional = True if (len(hps.state_dim) > 1) else False
    if 'ram' in hps.game:
        convolutional = False
    return convolutional

def model_description(var_list):
    ''' Print some model stats '''
    total_size = 0
    for v in var_list:
            total_size += np.prod([int(s) for s in v.get_shape()])
    return total_size

def split(x, split_dim, split_sizes):
    ''' Split tensor along split_dim in split_sizes '''
    n = len(list(x.get_shape()))
    dim_size = np.sum(split_sizes)
    assert int(x.get_shape()[split_dim]) == dim_size
    ids = np.cumsum([0] + split_sizes)
    ids[-1] = -1
    begin_ids = ids[:-1]

    ret = []
    for i in range(len(split_sizes)):
        cur_begin = np.zeros([n], dtype=np.int32)
        cur_begin[split_dim] = begin_ids[i]
        cur_end = np.zeros([n], dtype=np.int32) - 1
        cur_end[split_dim] = split_sizes[i]
        ret += [tf.slice(x, cur_begin, cur_end)]
    return ret

def repeat(x,k):
    ''' Repeat k times along first dimension '''
    def change(x,k):    
        shape = x.get_shape().as_list()[1:]
        x_1 = tf.expand_dims(x,1)
        tile_shape = tf.concat([tf.ones(1,dtype='int32'),[k],tf.ones([tf.rank(x)-1],dtype='int32')],axis=0)
        x_rep = tf.tile(x_1,tile_shape)
        new_shape = np.insert(shape,0,-1)
        x_out = tf.reshape(x_rep,new_shape)    
        return x_out
        
    return tf.cond(tf.equal(k,1),
                   lambda: x,
                   lambda: change(x,k))   
                   
def logsumexp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))
 
def make_copy_from_to(v1_list, v2_list):
    """
    Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
    The ordering of the variables in the lists must be identical.
    """
    v1_list = list(sorted(v1_list, key=lambda v: v.name))
    v2_list = list(sorted(v2_list, key=lambda v: v.name))
    
    update_ops = []
    for v1, v2 in zip(v1_list, v2_list):
        op = v2.assign(v1)
        update_ops.append(op)
    return tf.group(*update_ops, name='copy_op')
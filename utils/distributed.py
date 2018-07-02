# -*- coding: utf-8 -*-
"""
Helper functions for distributed tensorflow
@author: thomas
"""

import tensorflow as tf

def make_server(hps,cluster_spec):
    ''' set-up tensorflow cluster for distributed training '''
    if hps.distributed:
        config = tf.ConfigProto(
                device_filters=['/job:ps', '/job:worker/task:{}'.format(hps.job_index)], #/cpu:0
                #device_count={"CPU": hps.num_agents, "GPU" : 0},
                #allow_soft_placement=True,
                inter_op_parallelism_threads=2,
                intra_op_parallelism_threads=1,
                log_device_placement=False
                )       
        server = tf.train.Server(cluster_spec, 
                             config=config,
                             job_name="worker", 
                             task_index=hps.job_index)
    else:
        server,config = None,None    
    return server,config

def make_init_ops():
    ''' get initialization ops for distributed training '''
    global_variables = [v for v in tf.global_variables() if not v.name.startswith("local")]
    local_variables = [v for v in tf.global_variables() if v.name.startswith("local")]
    global_init_op = tf.variables_initializer(global_variables)
    local_init_op = tf.variables_initializer(local_variables)
    return global_init_op,local_init_op,global_variables
# -*- coding: utf-8 -*-
"""
Wrapper to call on slurm cluster
@author: thomas
"""

import tensorflow as tf
import os
import os.path
import re
import time
import numpy as np
import resource
import random

from hps import get_hps
#from duvn import agent
from duvn_agent import Agent

flags = tf.app.flags
flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
FLAGS = flags.FLAGS 

def main(_):
    hps = get_hps().parse(FLAGS.hpconfig) 
    if hps.agent_index == 0:
        print('Writing to {}'.format(hps.result_dir))
    print('Starting process {}'.format(hps.agent_index))

    if hps.slurm == False:
        raise ValueError('You should only call this script from setup.py to run on slurm cluster')         
    elif hps.distributed == False:
        Agent(hps)
    else:
        # Build the cluster
        cluster_spec,hps = make_cluster(hps)
        if hps.agent_index == 0:
            print('Agent {} sees cluster {}'.format(hps.agent_index,cluster_spec.as_cluster_def()))        
        #cluster = tf.train.ClusterSpec(cluster)
        if hps.job_type == 'ps':
            print('Starting parameter server {}'.format(hps.job_index))
            ps_function(hps,cluster_spec)
        else:
            print('Starting agent {}'.format(hps.job_index))
            Agent(hps,cluster_spec)

def ps_function(hps,cluster_spec):
    ps_servers = ["/job:ps/task:{}".format(ps_num) for ps_num in range(hps.n_ps)]
    config = tf.ConfigProto(
                device_filters=ps_servers, #/cpu:0
                #device_count={"CPU": hps.num_agents, "GPU" : 0},
                #allow_soft_placement=True,
                #inter_op_parallelism_threads=2,
                #intra_op_parallelism_threads=1,
                log_device_placement=False
                )     
    server = tf.train.Server(cluster_spec, 
                             config=config,
                             job_name="ps", 
                             task_index=hps.job_index)

    memory_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    print('Process {} = parameter server {} memory use {} Mb'.format(hps.agent_index,hps.job_index,memory_use))
    server.join()
        
def _pad_zeros(iterable, length):
    return (str(t).rjust(length, '0') for t in iterable)

def _expand_ids(ids):
    ids = ids.split(',')
    result = []
    for id in ids:
        if '-' in id:
            begin, end = [int(token) for token in id.split('-')]
            result.extend(_pad_zeros(range(begin, end+1), int(end-begin)))
        else:
            result.append(id)
    return result

def _expand_nodelist(nodelist):
    prefix, ids = re.findall("(.*)\[(.*)\]", nodelist)[0]
    ids = _expand_ids(ids)
    result = [prefix + str(id) for id in ids]
    return result

def merge_index(x,start,stop):
    new = [','.join(x[start:stop])]
    del x[(stop-1)]
    x[start] = new[0] 
    return x

def expand_nodelist(nodelist):
    nodelist = nodelist.split(',')
    #
    for i in reversed(range(len(nodelist))):
        if '[' in nodelist[i] and ']' in nodelist[i+1]:
            nodelist = merge_index(nodelist,i,i+2)
            #break
    for i in reversed(range(len(nodelist))):
        if '[' in nodelist[i]:
            expanded_node = _expand_nodelist(nodelist[i])   
            del nodelist[i]
            nodelist.extend(expanded_node) 
            break
    return nodelist

def sort_xy(x,*args):
    x_order = np.argsort(x)
    out = [x[x_order]]
    for arg in args:
        out.append(arg[x_order])
    return out

def get_new_port():
    while True:
        proposal = random.choice(np.arange(12000,13000))
    #for proposal in np.arange(12000,13000):
        return_val = os.system('netstat -an | grep {}'.format(proposal))
        if return_val == 256:
            break
    return proposal

def make_cluster(hps):
    #n_tasks = hps.n_ps + hps.n_agent
    nodelist = os.environ['SLURM_NODELIST']
    #print('nodelist = {}'.format(nodelist))
    #print('nodelist = {}'.format(nodelist))
    tasks_per_node = os.environ['SLURM_JOB_CPUS_PER_NODE']
    #tasks_per_node2 = os.environ['SLURM_TASKS_PER_NODE']
    #print('job cpu per node {}, tasks per node {}'.format(tasks_per_node,tasks_per_node2))
    #proc_id = os.environ['SLURM_PROCID']
    #nprocs = os.environ['SLURM_NPROCS']
    #nnodes = os.environ['SLURM_NNODES']
    #print('proc id {},nprocs {},nnodes {}'.format(proc_id,nprocs,nnodes))
    while True:
        #port = os.environ['SLURM_STEP_RESV_PORTS']
        port = get_new_port()
            
        #print('port = {}'.format(port))        
        own_node = os.environ['SLURMD_NODENAME']
        #print(hps.agent_index,own_node,port)

#        if own_node == 'ess' or own_node == 'grs':
#            own_node += '-{}'.format(port)
#            port = get_new_port()
            
            # need to find new port
        with open(hps.result_dir + 'ports.txt','a') as file:
            file.write('{}+{}+{}\n'.format(hps.agent_index,own_node,port))
            
        #print('Agent {} wrote {},{} to ports.txt'.format(hps.agent_index,own_node,port))
    
        # wait for all processes to write:
        num_agents = hps.n_agent + hps.n_ps
        while count_file_size(hps.result_dir + 'ports.txt') < num_agents:
            time.sleep(1)
            #print('Agent {}, size of ports.txt = {}, still waiting'.format(hps.agent_index,count_file_size(hps.result_dir + 'ports.txt')))

        # read back file:
        lines = read_file(hps.result_dir + 'ports.txt')
        agent_index,node_index,port_index = get_agent_list(lines)
    
        if (hps.agent_index==0):
            if len(port_index) == len(np.unique(port_index)):
                with open(hps.result_dir + 'correct.txt','w') as file:
                    file.write('found a cluster\n')
                jump_out = True
                print('Found a cluster')
                break
            else:
                # try again
                jump_out = False
                open(hps.result_dir + 'ports.txt', 'w').close() # clear the file
                #print('Cleared the file')
        else:
            while True:
                if os.path.exists(hps.result_dir + 'correct.txt'):
                    # we may exit
                    jump_out = True
                    #print('Found correct.txt for agent {}'.format(hps.agent_index))
                    break
                elif (not os.path.exists(hps.result_dir + 'ports.txt')):
                    # need a new iteration, as agent 0 removed the ports.txt file
                    jump_out = False
                    #print('Could not find ports.txt for agent {}'.format(hps.agent_index))
                    break
                else:
                    continue
        if jump_out:
            break
                
            
    #print(agent_index,node_index,port_index)
    
    #print(tasks_per_node)
    #nodelist = expand_nodelist(nodelist)
    #tasklist = proc_tasks(tasks_per_node)          
    #tasklist = [int(x/2) for x in tasklist]
    
    #print('node list {}, tasks per node {}'.format(nodelist,tasklist))

    # Build the cluster
    #cluster,hps = make_dict(nodelist,tasklist,hps,port)
   
    cluster,hps = make_dict2(hps,agent_index,node_index,port_index)
    # print(cluster)
    return cluster,hps

def proc_tasks(tasks_per_node):
    tasks_per_node = tasks_per_node.split(',')
    out = []
    for task in tasks_per_node:
        if '(' in task:
            t1,t2 = task.split('(')
            out.extend([int(t1[0])]*int(t2[1]))
        else:
            out.extend([int(task)])
    return out

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def get_agent_list(lines):
    lines = [line.split('+') for line in lines]
    agent_index = np.array([int(x[0]) for x in lines])
    node_index = np.array([x[1] for x in lines])
    port_index = np.array([int(x[2]) for x in lines])
    agent_index,node_index,port_index = sort_xy(agent_index,node_index,port_index)
    return agent_index,node_index,port_index

def read_file(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    return content

def count_file_size(path):
    with open(path) as f:
        val = sum(1 for _ in f)
    return val

def make_dict(nodelist,tasklist,hps,port):
    ps = []
    workers = []
    count = 0
    for i,node in enumerate(nodelist):
        port = 12100
        for j in range(tasklist[i]):
            if count >= hps.n_agent:
                ps.append('{}:{}'.format(node,port))
                if count == hps.agent_index:
                    hps.job_type = 'ps'
                    hps.job_index = count - hps.n_agent
            else:
                workers.append('{}:{}'.format(node,port))
                if count == hps.agent_index:
                    hps.job_type = 'worker'
                    hps.job_index = count
            port += 1
            count += 1
    cluster = {"worker": workers, "ps" : ps}
    return cluster,hps

def make_dict2(hps,agent_index,node_index,port_index):
    ps = []
    workers = []
    count = 0
    for i in range(hps.n_ps + hps.n_agent):
        if i >= hps.n_agent:
            ps.append('{}:{}'.format(node_index[i],port_index[i]))
            if count == hps.agent_index:
                hps.job_type = 'ps'
                hps.job_index = count - hps.n_agent
        else:
            workers.append('{}:{}'.format(node_index[i],port_index[i]))
            if count == hps.agent_index:
                hps.job_type = 'worker'
                hps.job_index = count
        count += 1
    cluster_list = {"worker": workers, "ps" : ps}
    cluster_spec = tf.train.ClusterSpec(cluster_list)#.as_cluster_def()
    return cluster_spec,hps

if __name__ == "__main__":   
    tf.app.run()
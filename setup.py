# -*- coding: utf-8 -*-
"""
Setup script that loops over hyperparameters and repetitions.
Either submits to slurm, or runs locally
@author: thomas
"""
# Plotting setup
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#plt.style.use('ggplot')
#mpl.rcParams['lines.linewidth'] = 5    

import tensorflow as tf
import numpy as np
import os
import logging
import time
import shutil
import copy
from pprint import pformat

#from duvn import agent
from duvn_agent import Agent
from hps import get_hps
from utils.pytils import make_logger
np.set_printoptions(threshold=np.nan)

flags = tf.app.flags
flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
FLAGS = flags.FLAGS 

def import_matplotlib(hps):
    global mpl
    import matplotlib as mpl
    if hps.slurm:
        mpl.use('Agg')
    else:
        mpl.rcParams['lines.linewidth'] = 5
        mpl.rcParams.update({'font.size': 14})
        mpl.rcParams['axes.facecolor']='white'
        mpl.rcParams['savefig.facecolor']='white'
    global plt
    import matplotlib.pyplot as plt
    #plt.style.use('ggplot')
    plt.style.use('fivethirtyeight')
    plt.rcParams['lines.linewidth'] = 5
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'
    
def run(hps):
    'Main loop'
    logger = logging.getLogger('root')
    overall_begin = time.time()
    seq1,seq2,seq3,seq4,n_rep = hps.seq1,hps.seq2,hps.seq3,hps.seq4,hps.n_rep   

    for it1,item1 in enumerate(seq1):
        if hps.item1 is not None: hps._set(hps.item1,item1)
        for it2,item2 in enumerate(seq2):
            if hps.item2 is not None: hps._set(hps.item2,item2)
            for it3,item3 in enumerate(seq3):
                if hps.item3 is not None: hps._set(hps.item3,item3)
                for it4,item4 in enumerate(seq4):
                    if hps.item4 is not None: hps._set(hps.item4,item4)
                    for rep in range(n_rep):    
                        hps.result_dir = hps.base_result_dir + 'subplots/'
                        if hps.loop_hyper: hps.result_dir = hps.result_dir + make_name('',hps.item1,item1,hps.item2,item2,hps.item3,item3,hps.item4,item4) + '/' 
                        hps.rep = rep
                        hps.result_dir += 'rep:{}'.format(rep) + '/'

                        if not os.path.exists(hps.result_dir):
                            os.makedirs(hps.result_dir)
                        
                        if 'thompson' in hps.policy:
                            try:
                                policy,n_thompson_sample = hps.policy.split('-')    
                                hps.policy = policy
                                hps.n_thompson_sample = int(n_thompson_sample)
                            except:
                                pass
                        
                        if 'dropout' in hps.uncer:
                            try:
                                uncer,p_dropout = hps.uncer.split('-')    
                                hps.uncer = uncer
                                hps.p_dropout = float(p_dropout)
                            except:
                                pass
                    
                        if 'vi' in hps.uncer:
                            try:
                                uncer,sigma_prior = hps.uncer.split('-')    
                                hps.uncer = uncer
                                hps.sigma_prior = float(sigma_prior)
                            except:
                                pass

                        if 'lin_bay' in hps.uncer:
                            try:
                                uncer,sigma_prior = hps.uncer.split('-')    
                                hps.uncer = uncer
                                hps.sigma_prior = float(sigma_prior)
                            except:
                                pass
                                                    
                        # Launch the agent script
                        if hps.slurm:
                            # build a slurm submission
                            hps.visualize = False
                            sub_name = make_name('',hps.item1,item1,hps.item2,item2,hps.item3,item3,hps.item4,item4)
                            submit_slurm(hps,sub_name)  
                        else:
                            # run the process locally, and wait for it to finish
                            tf.reset_default_graph()
                            Agent(hps)
                            
    if not hps.slurm: 
        logger.info('Finished training, total time {} hours'.format((time.time()-overall_begin)/3600))
    
def submit_slurm(hps,sub_name):
    logger = logging.getLogger('root')
    # make the sh file
    run_name = 'jobs/batch_scripts/scr_{}_{}_{}.sh'.format(hps.game,sub_name,hps.rep)
    
    if hps.distributed:
        ntasks = hps.n_agent + hps.n_ps
        nodes = '1-3'
        n_cpu = hps.cpu_per_task   
        mem_per_cpu = int((16384/(ntasks*n_cpu)) - 5)
    else:
        ntasks = 1
        nodes = 1
        n_cpu = hps.cpu_per_task   
        mem_per_cpu = hps.mem_per_cpu
        
    base = 'srun python3 duvn_slurm.py' if not hps.distributed else 'srun -N1 -n1 python3 duvn_slurm.py' # --exclusive --resv-ports=1
    
    with open(run_name,'w') as fp:
        fp.write('#!/bin/sh\n')    
        fp.write("echo 'SLURM_NODELIST' $SLURM_NODELIST\n")
        fp.write("echo 'SLURM_JOB_CPUS_PER_NODE' $SLURM_JOB_CPUS_PER_NODE\n")
        
        for i in reversed(range(ntasks)):
            hps.agent_index = i
            fp.write(base + ' --hpconfig {}'.format(hps_to_list(hps)))
            if hps.distributed: fp.write(' &\n')
        if hps.distributed: fp.write('\nwait')
            
    # call sbatch
    cwd = os.getcwd()
    my_sbatch = ' '.join(['sbatch --partition=general --qos={} --time={} --ntasks={}',
                '--nodes={} --cpus-per-task={} --mem-per-cpu={} --mail-type=NONE',
                '--output=results/slurmout/slurm-%j.out',
                '--exclude=ess-2',
                #'--error=results/slurmout/eslurm-%j.out',
                '--workdir={}',
                '--job-name={} {}']).format(hps.slurm_qos,hps.slurm_time,ntasks,nodes,n_cpu,mem_per_cpu,cwd,hps.game[0:3]+hps.game[-1] + '_' + hps.name,run_name)
    logger.info('Starting experiment {}'.format(hps.name)) 
    logger.info(my_sbatch)
    os.system('chmod +x {}'.format(run_name))
    return_val = os.system(my_sbatch)
    if return_val != 0:
        raise ValueError('submission went wrong')

def hps_to_list(hps):
    out=[]
    hps_dict = copy.deepcopy(hps.__dict__)
    try:
        del hps_dict['_items']
    except:
        pass
    for k,v in hps_dict.items():
        if type(v) == list:
            v='+'.join(str(x) for x in v)
        if not (v is None or v == 'None'): # should not write the default hyperloop settings
            out.append('{}={}'.format(k,v))
    out.sort()
    return ','.join(out)

def hps_to_dict(hps):
    hps_dict = copy.deepcopy(hps.__dict__)
    try:
        del hps_dict['_items']
    except:
        pass
    return hps_dict

def get_hyperloop_name(hps):
    name = ''
    if hps.item1 is not None:
        name += '{}'.format(hps.item1)
    if hps.item2 is not None:
        name += '_{}'.format(hps.item2)
    if hps.item3 is not None:
        name += '_{}'.format(hps.item3)
    if hps.item4 is not None:
        name += '_{}'.format(hps.item4)
    return name

def make_name(basename='',item1=None,val1=None,item2=None,val2=None,item3=None,val3=None,item4=None,val4=None,separator='-'):
    name = basename
    if item1 is not None:
        name += '{}:{}'.format(item1,val1)
    if item2 is not None:
        name += separator + '{}:{}'.format(item2,val2)
    if item3 is not None:
        name += separator + '{}:{}'.format(item3,val3)
    if item4 is not None:
        name += separator + '{}:{}'.format(item4,val4)
    return name

def make_result_dir(hps):   
    name = os.getcwd() + '/results' 
    name += '/game:{}'.format(hps.game) + '/'
    if hps.name is not 'None':
        name += hps.name 
    else:
        if hps.loop_hyper:
            name += 'hyperloop_{}'.format(get_hyperloop_name(hps))
        else:
            name += 'solo_{}_{}'.format(hps.output,hps.policy)
            if hps.p_dropout < 1.0:
                name += '_dropout'
    i = 0
    while os.path.exists(name + '/{0:04}/'.format(i)) or os.path.exists(name + '/{0:04}d/'.format(i)):
        i += 1
    name += '/{0:04}/'.format(i)
    result_dir = name
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    checkpoint_dir = '/tmp' + name
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir)
    return result_dir,checkpoint_dir

def main(_):
    'Set-up training'
    # parsing 
    hps = get_hps().parse(FLAGS.hpconfig)
    import_matplotlib(hps)
    
    # Logging and saving    
    hps.base_result_dir,hps.checkpoint_dir = make_result_dir(hps)
    logger = make_logger(hps.base_result_dir,name='root',level=hps.level)
    logger.info('Created new base results folder at: {}'.format(hps.base_result_dir))
    logger.info('Starting experiment {} on environment {}'.format(hps.name,hps.game))

    # Write hyperparameters    
    with open(hps.base_result_dir + 'hps.txt','w') as file:
        file.write(pformat(hps_to_dict(hps)))
    with open(hps.base_result_dir + 'hps_raw.txt','w') as file:
        file.write(hps_to_list(hps))
    run(hps)

if __name__ == "__main__":   
    tf.app.run()

# -*- coding: utf-8 -*-
"""
Repeat a set of tasks over some hyperparameters
@author: thomas
"""
import numpy as np
import sys

def pygame_setup():
    games = ['MonsterKong-v0','Catcher-v0','FlappyBird-v0','PuckWorld-v0','RaycastMaze-v0','Snake-v0']
    eps = [50000]*6
    job_type = 5*['short']# + 2 * ['long']
    job_time = 5*['0-03:59:00']# + 2 * ['4-12:00:00']
    n_bins = [5]*5
    bin_min = [0.0]*5
    bin_max = [1.0]*5  
    sd_output = 5 * [1.0]
    return games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output  

def chain_setup():
    games = ['Chain-10','Chain-25','Chain-50','Chain-100']
    n = len(games)
    eps = n*[3000]
    job_type = n*['short']# + 2 * ['long']
    job_time = n*['0-03:59:00']# + 2 * ['4-12:00:00']
    n_bins = n*[5]
    bin_min = n*[-0.1]
    bin_max = n*[1.1]
    sd_output = n*[0.5]
    return games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output  

def chain_ordered_setup():
    games = ['ChainOrdered-10','ChainOrdered-30','ChainOrdered-50','ChainOrdered-100']
    n = len(games)
    eps = n*[3000]
    job_type = n*['short']# + 2 * ['long']
    job_time = n*['0-03:59:00']# + 2 * ['4-12:00:00']
    n_bins = n*[5]
    bin_min = n*[0.0]
    bin_max = n*[1.0]
    sd_output = n*[1.0]
    return games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output  

def atari_ram2_setup():
    games = ['DemonAttack-ram-v0','Enduro-ram-v0','Kangaroo-ram-v0','Riverraid-ram-v0','Seaquest-ram-v0','UpNDown-ram-v0']
    n = len(games)
    eps = n * [1000000]
    job_type = n * ['short']
    job_time = n * ['0-03:59:00']
    n_bins = n * [51]
    bin_min = [-25] + (n-1) * [-15]
    bin_max = [25] + (n-1) * [15]
    sd_output = n * [1.0]
    return games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output  

def atari_ram_setup():
#    games = ['Pong-ram-v0','Amidar-ram-v0','Assault-ram-v0','Asteroids-ram-v0','BattleZone-ram-v0','Atlantis-ram-v0']
    games = ['Amidar-ram-v0','Asteroids-ram-v0']
    n = len(games)
    eps = n * [1000000]
    job_type = n * ['short']
    job_time = n * ['0-03:59:00']
    n_bins = n * [51]
    bin_min = n * [-25]
    bin_max = n * [25]
    sd_output = n * [1.0]
    return games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output  

def control_setup_r():
    games = ['CartPole-vr','MountainCar-vr','Acrobot-vr','LunarLander-vr','FrozenLakeNotSlippery-v0','FrozenLakeNotSlippery-v1']
    n = len(games)    
    eps = n * [10000]
    job_type = n * ['short']
    job_time = n * ['0-03:59:00']
    n_bins = n * [31]
    bin_min = n * [-1.2]
    bin_max = n * [1.2]
    sd_output = n * [1.0]
    return games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output  

def control_setup():
    games = ['CartPole-v0','MountainCar-v0','Acrobot-v1','LunarLander-v2']
    n = len(games)    
    eps = n * [10000]
    job_type = n * ['short']
    job_time = n * ['0-03:59:00']
    n_bins = n * [31]
    bin_min = [0,-200,-200,-300]
    bin_max = [200,0,0,300]
    sd_output = n * [1.0]
    return games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output  
    
def atari_conv_setup():
    games = ['Breakout-v0','Pong-v0','MontezumaRevenge-v0']
    eps = [10000000]*3
    job_type =  3 * ['long']
    job_time =  3 * ['2-12:00:00']
    n_bins = [51] * 3
    bin_min = [-15,-15,-15]
    bin_max = [15,15,15]
    sd_output = 3 * [1.0]
    return games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output  

def distributed_setup(n,distributed,n_agent=16,n_ps=2):
    if distributed:
        distr_bool = [True]*n
    else:
        distr_bool = [False]*n
    n_pss = [n_ps]*n
    n_agents = [n_agent]*n
    return distr_bool,n_pss,n_agents

def expand(job,task,distributed):
    t_max = 10000000
    print(task)
    if task == 'chain':
        games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output = chain_setup()
    if task == 'chain_ordered':
        games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output = chain_ordered_setup()
    elif task == 'control':
        games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output = control_setup()  
    elif task == 'control_r':
        games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output = control_setup_r()  
    elif task == 'atari_ram':
        games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output = atari_ram_setup()          
    elif task == 'atari_ram2':
        games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output = atari_ram2_setup()          
    elif task == 'atari_conv':
        games,eps,job_type,job_time,n_bins,bin_min,bin_max,sd_output = atari_conv_setup() 
    
    distr_bool,n_pss,n_agents = distributed_setup(len(games),distributed)    
    
    with open(job+'.sh') as fp:
        lines = fp.readlines()
    lines = [x.strip() for x in lines]
    
    with open(job+'_' + task + '.sh','w') as fp:
        for i,game in enumerate(games):
            fp.write('#game={} and max_ep={}'.format(game,eps[i]))
            fp.write('\n')
            for line in lines:
                if line is not '' and '#' not in line:
                    subline = line.split()
                    subline[3] = subline[3] + ',game={},max_ep={},t_max={},slurm_qos={},slurm_time={},slurm=True,level=error,n_bins={},cat_min={},cat_max={},sd_output_bias={},distributed={},n_ps={},n_agent={}'.format(games[i],eps[i],t_max,job_type[i],job_time[i],n_bins[i],bin_min[i],bin_max[i],sd_output[i],distr_bool[i],n_pss[i],n_agents[i]) 
                    ## WATCH OUT: for now distributed training always without replay due to memory limits
                    if distributed:
                        subline[3] += ',cpu_per_task=1'
                        
                    fp.write(' '.join(subline))
                    fp.write('\n')
            fp.write('\n')
        
if __name__ == "__main__":   
    job = 'job_basic' if len(sys.argv) < 2 else sys.argv[1]
    task = 'chain' if len(sys.argv) < 3 else sys.argv[2]
    distributed = False if len(sys.argv) < 4 else (sys.argv[3] == 'True')
    print('Start expanding '.format(job))
    expand(job,task,distributed)
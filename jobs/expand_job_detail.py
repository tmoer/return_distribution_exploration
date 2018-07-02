# -*- coding: utf-8 -*-
"""
Repeat a set of tasks over some hyperparameters
@author: thomas
"""

games = ['Chain','Taxi','CartPole-v0']
eps = 3*[10000]
job_type = 3*['short']
job_time = 3*['3:59:00']

with open('job_detail.sh') as fp:
    lines = fp.readlines()
lines = [x.strip() for x in lines]

with open('jobs_collected_detail.sh','w') as fp:
    for i,game in enumerate(games):
        fp.write('#game={} and max_ep={}'.format(game,eps[i]))
        fp.write('\n')
        for line in lines:
            if line is not '' and '#' not in line:
                subline = line.split()
                subline[3] = subline[3] + ',game={},max_ep={},slurm_qos={},slurm_time={},slurm=True,level=warning'.format(games[i],eps[i],job_type[i],job_time[i]) 
                fp.write(' '.join(subline))
                fp.write('\n')
        fp.write('\n')

# -*- coding: utf-8 -*-
"""
Script to automatically determine how many processes to request
@author: thomas
"""

import os
import sys

def submit(path):
    with open(path) as fp:
        lines = fp.readlines()
    lines = [x.strip() for x in lines]
    
    for line in lines:
        if line is not '' and '#' not in line:
            os.system(line)

if __name__ == "__main__":   
    job_lists = ['job_basic_chain.sh'] if len(sys.argv) < 2 else sys.argv[1:]
    print(job_lists)
    for job_list in job_lists:
        path = './jobs/'+ job_list
        print('Start submitting from file {}'.format(path))
        submit(path)    
        print('Done')                   

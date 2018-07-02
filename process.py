# -*- coding: utf-8 -*-
"""
Process result directories

@author: thomas
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from cycler import cycler
plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 5
lines = ["-","--","-.",":"]
colours = ['r','g','b','y','k','w']
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y','c','m','k','w']*2) +
                           cycler('linestyle', [i for i in ['-', '--'] for j in range(8)])))
import os
import numpy as np
import sys
import shutil
from hps import get_hps

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

def get_subdirs(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def process(base_result_dir,overview_dir,rep_dir):
    print('Processing folder {}'.format(base_result_dir))

    # load hps associated with this folder
    try: 
        with open(base_result_dir + '/hps_raw.txt', 'r') as f:
            hps_list = f.read()        
        hps = get_hps().parse(hps_list)
    except Exception as e:
        print('Base experiment folder {} with error {}'.format(base_result_dir,e))
        print('Removing empty folder {}'.format(base_result_dir))
        shutil.rmtree(base_result_dir)
        return
        
    # Set-up plots        
    ep_run_plot = xy_subplot(xlabel='episodes',ylabel='Episode reward',row_item=hps.item4,row_seq=hps.seq4,col_item=hps.item3,col_seq=hps.seq3)
#    av_run_plot = xy_subplot(xlabel='episodes',ylabel='Average reward',row_item=hps.item4,row_seq=hps.seq4,col_item=hps.item3,col_seq=hps.seq3)    
#    Qsa_run_plot = xy_subplot(xlabel='episodes',ylabel='Qsa norm',row_item=hps.item4,row_seq=hps.seq4,col_item=hps.item3,col_seq=hps.seq3)    
#    grad_run_plot = xy_subplot(xlabel='episodes',ylabel='Gradient norm',row_item=hps.item4,row_seq=hps.seq4,col_item=hps.item3,col_seq=hps.seq3)    
#    loss_run_plot = xy_subplot(xlabel='episodes',ylabel='Loss',row_item=hps.item4,row_seq=hps.seq4,col_item=hps.item3,col_seq=hps.seq3)    

    # load in all data
    all_empty = True
    for it1,item1 in enumerate(hps.seq1):
        for it2,item2 in enumerate(hps.seq2):
            for it3,item3 in enumerate(hps.seq3):
                for it4,item4 in enumerate(hps.seq4):                        
                    result_dir = base_result_dir + '/subplots/'
                    if hps.loop_hyper:
                        result_dir += make_name('',hps.item1,item1,hps.item2,item2,hps.item3,item3,hps.item4,item4) + '/'

                    if not os.path.exists(result_dir):
                        continue
                    
                    #ep_R_plot = xy_plot(ylabel='Episode reward',xlabel='episodes')
                    #av_R_plot = xy_plot(ylabel='Average reward',xlabel='episodes')
                    ep_c,ep_R_c,av_R_c,Qsa_c,grad_c,loss_c = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

                    for rep in range(hps.n_rep):
                        read_dir = result_dir + 'rep:{}'.format(rep) + '/'
                        if not os.path.exists(read_dir):
                            continue

                        # Load raw data
                        try:
                            eps = np.loadtxt(read_dir+'episode_raw.txt')                    
                            ep_R = np.loadtxt(read_dir+'ep_reward_raw.txt')

                            if (len(eps)>0) and (len(ep_R)>0):
                                # sometimes a txt get accidently empty due to a time limit when writing
                                ep_c = np.append(ep_c,eps)
                                ep_R_c = np.append(ep_R_c,ep_R)

                            all_empty = False

                            #av_R = np.loadtxt(read_dir+'av_reward_raw.txt')
                            #Qsa_norm = np.loadtxt(read_dir+'Qsa_norm_raw.txt')
                            #grad_norm = np.loadtxt(read_dir+'grad_norm_raw.txt')
                            #loss = np.loadtxt(read_dir+'loss_raw.txt')
                        except Exception as e:
                            print(e)
                            continue
                    
                            #av_R_c = np.append(av_R_c,av_R)
                            #Qsa_c = np.append(Qsa_c,Qsa_norm)
                            #grad_c = np.append(grad_c,grad_norm)
                            #loss_c = np.append(loss_c,loss)

                        
                    # Finish repetition plots
                    #ep_R_plot.finish()
                    #av_R_plot.finish()
                    #ep_R_plot.save(result_dir+'episode_reward')
                    #av_R_plot.save(result_dir+'average_reward')

                    if len(ep_c) == 0 or len(ep_R_c) == 0:
                        print('empty {}, skipping folder'.format(base_result_dir))
                        ep_c = np.array([0,1])
                        ep_R_c = np.array([0,0])
                        
                    # Do smoothing over repetitions
                    ep_run,ep_R_run = downsample_smooth(ep_c,ep_R_c,down_len=1000,window=50)
                    
                    np.savetxt(result_dir+'episode_run.txt',ep_run,fmt='%.3g')                    
                    np.savetxt(result_dir+'ep_reward_run.txt',ep_R_run,fmt='%.3g') 

                    label = make_name('',hps.item1,item1,hps.item2,item2)
                    
                    max_ep = 10000
                    ep_R_run = ep_R_run[ep_run<max_ep]
                    ep_run = ep_run[ep_run<max_ep]                    
                    ep_run_plot.add(x=ep_run,y=ep_R_run,row=it4,col=it3,label=label) 
                        
#                    try:                        
#                        ep_run,ep_R_run,av_R_run,Qsa_run,grad_run,loss_run = downsample_smooth(ep_c,ep_R_c,av_R_c,Qsa_c,grad_c,loss_c,down_len=1000,window=50)
#                        np.savetxt(result_dir+'av_reward_run.txt',av_R_run,fmt='%.3g')
#                        np.savetxt(result_dir+'Qsa_run.txt',Qsa_run,fmt='%.3g') 
#                        np.savetxt(result_dir+'grad_run.txt',grad_run,fmt='%.3g') 
#                        np.savetxt(result_dir+'loss_run.txt',loss_run,fmt='%.3g') 
#
#
#                        # add to the higher level plot
#                        label = make_name('',hps.item1,item1,hps.item2,item2)
#                        av_run_plot.add(x=ep_run,y=av_R_run,row=it4,col=it3,label=label)
#                        Qsa_run_plot.add(x=ep_run,y=Qsa_run,row=it4,col=it3,label=label)
#                        grad_run_plot.add(x=ep_run,y=grad_run,row=it4,col=it3,label=label)
#                        loss_run_plot.add(x=ep_run,y=loss_run,row=it4,col=it3,label=label)      
#                    except:
#                        pass
                    #try:
                    #    Qsa_run = downsample_smooth(ep_c,ep_R_c,av_R_c,down_len=1000,window=50)
                    #    np.savetxt(result_dir+'Qsa_norm_run.txt',Qsa_run,fmt='%.3g') 
                    #    Qsa_run_plot.add(x=ep_run,y=Qsa_run,row=it4,col=it3,label=label)  
                    #except:
                    #    pass
    if all_empty:
        print('Removing empty folder {}'.format(base_result_dir))
        shutil.rmtree(base_result_dir)
    else:
        ep_run_plot.finish()
        ep_run_plot.save(base_result_dir+'/Episode_reward_running',close=False)
        ep_run_plot.save(overview_dir+'Episode_reward/{}_{}_{}'.format(hps.game,hps.name,rep_dir))
#        av_run_plot.finish()
#        av_run_plot.save(base_result_dir+'/Average_reward_running',close=False)
#        av_run_plot.save(overview_dir+'Average_reward/{}_{}_{}'.format(hps.game,hps.name,rep_dir))
#        Qsa_run_plot.finish()
#        Qsa_run_plot.save(base_result_dir+'/Qsa_norm_running',close=False)
#        Qsa_run_plot.save(overview_dir+'Qsa_norm/{}_{}_{}'.format(hps.game,hps.name,rep_dir))
#        grad_run_plot.finish()
#        grad_run_plot.save(base_result_dir+'/grad_norm_running',close=False)
#        grad_run_plot.save(overview_dir+'grad_norm/{}_{}_{}'.format(hps.game,hps.name,rep_dir))
#        loss_run_plot.finish()
#        loss_run_plot.save(base_result_dir+'/loss_running',close=False)
#        loss_run_plot.save(overview_dir+'loss/{}_{}_{}'.format(hps.game,hps.name,rep_dir))


    # Mark this folder as processed
    #os.rename(base_result_dir,base_result_dir+'d')
    print('Processed folder')

def plot_result_array(result_array,result_name=None,item1=None,seq1=[None],item2=None,seq2=[None],item3=None,seq3=[None],item4=None,seq4=[None]):
    ''' result_array expects a single result in each entry in the (up to) 4D array'''

    #while result_array.ndim < 4:
    #    result_array = np.expand_dims(result_array,-2)
    item_len = result_array.shape   
    x_range = np.arange(1,item_len[0]+1)
    col_size = item_len[2]
    row_size = item_len[3]  
    
    fig,ax = plt.subplots(nrows=row_size,ncols=col_size,sharex=True,sharey=True)
    fig.set_figheight(row_size*7)
    fig.set_figwidth(col_size*7)
    for k in range(row_size):
        for j in range(col_size):
            try:
                if ax.ndim == 2:
                    col = ax[k,j]
                elif ax.ndim == 1:
                    col = ax[j]
            except:
                col = ax
            for l in range(item_len[1]):
                if item2 is not None:
                    col.plot(x_range,result_array[:,l,j,k],markersize=10,mew=5,marker='x',label='{} = {}'.format(item2,seq2[l]))
                else:
                    col.plot(x_range,result_array[:,l,j,k],markersize=10,mew=5,marker='x')                    
            if item3 is not None and item4 is not None:
                col.set_title('{} = {}, {} = {}'.format(item3,seq3[j],item4,seq4[k]),fontsize=15)
            elif item3 is not None:
                col.set_title('{} = {}'.format(item3,seq3[j]),fontsize=15)
            if item1 is not None:
                col.set_xlabel(item1,fontsize=15)
            if result_name is not None:
                col.set_ylabel(result_name,rotation=90,fontsize=15)   
            col.set_xticks(x_range)
            col.set_xlim([x_range[0]-0.3,x_range[-1]+0.3])
            col.set_xticklabels(seq1,fontsize=15)
            #col.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
            col.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
            #format_ticks(col,'x')
            #format_ticks(col,'y')
            col.legend(loc=0,fontsize=15)
    fig.tight_layout()

def downsample_smooth(ep_c,*args,down_len=500,window=50):
    # Sort
    out = sort_xy(ep_c,*args)
    
    # Downsample
    for i in range(len(out)):
        out[i] = downsample(out[i],down_len)

    # Smooth
    if len(out[0]) < window: 
        window = 1 # only for debugging, can't average short curves
    for i in range(1,len(out)):
        out[i] = running_mean(out[i],window)
    out[0] = out[0][(window-1):]      
    
    return out

def downsample(x,out_len=1000):
    # recursively downsample array x by halving it
    # The out_len will be between out_len and 2*out_len
    if x.ndim == 0:
        x = x[None]
    elif x.ndim>1:
        x = np.squeeze(x)
    while len(x)>2*out_len:
        if is_odd(len(x)):
            x = x[:-1] # remove last item
        x = np.squeeze(np.mean(np.reshape(x,[-1,2]),axis=1))
    return x

def running_mean(x,window=50):
    return np.convolve(x, np.ones((window,))/window, mode='valid')

def is_odd(num):
    return bool(num & 1)

def sort_xy(x,*args):
    x = x[:-1]
    x_order = np.argsort(x)
    out = [x[x_order]]
    for arg in args:
        out.append(arg[x_order])
    return out

class xy_plot():
    def __init__(self,title=None,xlabel=None,ylabel=None):
        self.fig,self.ax=plt.subplots()
        if title is not None: self.ax.set_title(title)
        if xlabel is not None: self.ax.set_xlabel(xlabel)
        if ylabel is not None: self.ax.set_ylabel(ylabel)

    def add(self,x,y):
        self.ax.plot(x,y)
    
    def finish(self):
        self.ax.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
        self.ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
        self.ax.autoscale()
        self.fig.tight_layout()

    def save(self,save_dir):
        plt.figure(self.fig.number)
        plt.savefig('{}.png'.format(save_dir),bbox_inches="tight")
        plt.close()

class xy_subplot():
    def __init__(self,xlabel=None,ylabel=None,row_item=None,row_seq=[None],col_item=None,col_seq=[None]):
        self.row_size = row_size = len(row_seq)
        self.col_size = col_size = len(col_seq)
        self.fig,self.ax=plt.subplots(nrows=row_size,ncols=col_size,sharex=True,sharey=True)
        self.fig.set_figheight(row_size*9)
        self.fig.set_figwidth(col_size*9)
        for i in range(row_size):
            for j in range(col_size):
                try:
                    if self.ax.ndim == 2:
                        col = self.ax[i,j]
                    elif self.ax.ndim == 1:
                        col = self.ax[j]
                except:
                    col = self.ax
                if xlabel is not None:
                    col.set_xlabel(xlabel,fontsize=15)
                if ylabel is not None:
                    col.set_ylabel(ylabel,fontsize=15)
                if row_item is not None and col_item is not None:
                    col.set_title('{} = {}, {} = {}'.format(col_item,col_seq[j],row_item,row_seq[i]),fontsize=15)
                elif col_item is not None:
                    col.set_title('{} = {}'.format(col_item,col_seq[j]),fontsize=15)

    def add(self,x,y=None,row=0,col=0,label=''):
        try:
            if self.ax.ndim == 2:
                col = self.ax[row,col]
            elif self.ax.ndim == 1:
                col = self.ax[col]
        except:
            col = self.ax
            
        if y is not None:
            col.plot(x,y,label=label,linewidth=5)
        else:
            col.plot(x,label=label,linewidth=5)
    
    def finish(self):
        for i in range(self.row_size):
            for j in range(self.col_size):
                try:
                    if self.ax.ndim == 2:
                        col = self.ax[i,j]
                    elif self.ax.ndim == 1:
                        col = self.ax[j]
                except:
                    col = self.ax
                #format_ticks(col,'x')
                #format_ticks(col,'y')
                col.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
                col.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
                y_lim = list(col.get_ylim())
                y_lim[0] -= (y_lim[1]-y_lim[0])/50
                y_lim[1] += (y_lim[1]-y_lim[0])/50                
                col.set_ylim(y_lim)
                if (j+1) == self.col_size and (i == 0):                                    
                    col.legend(loc='upper left',bbox_to_anchor=(1.04,1),ncol=1,fontsize=15)
        self.fig.tight_layout()

    def save(self,save_dir,close=True):
        plt.figure(self.fig.number)
        plt.savefig('{}.png'.format(save_dir),bbox_inches="tight")
        if close:        
            plt.close()
        
def format_ticks(ax,axis='x'):
    #ax.autoscale()
    ticks = ax.get_xticks().tolist() if axis == 'x' else ax.get_yticks().tolist()
    #ticks = [format(tick,'.3g') for tick in ticks]
    ax.set_xticklabels(ticks,fontsize=9) if axis == 'x' else ax.set_yticklabels(ticks,fontsize=9)

def loop_directories(result_dir,overview_dir):
    sub_dirs = get_subdirs(result_dir)
    for game_dir in sub_dirs:
        if 'game' in game_dir:
            name_dirs = get_subdirs(result_dir + game_dir + '/' )
            for name_dir in name_dirs:
                rep_dirs = get_subdirs(result_dir + game_dir + '/' + name_dir + '/')
                for rep_dir in rep_dirs:
                    if not 'd' in rep_dir:
                        # Need to process this one
                        process(result_dir + game_dir + '/' + name_dir + '/' + rep_dir,overview_dir,rep_dir)

if __name__ == "__main__":   
    folder = 'icml_results' if len(sys.argv) < 2 else sys.argv[1]
    result_folder = '/home/thomas/' + folder + '/'
    print('Start looping through result folder {}'.format(result_folder))
    overview_dir= result_folder+'learning_curves/'
    if not os.path.exists(overview_dir):
        os.makedirs(overview_dir)
        os.makedirs(overview_dir + 'Episode_reward/')
        os.makedirs(overview_dir + 'Average_reward/')
        os.makedirs(overview_dir + 'Qsa_norm/')
        os.makedirs(overview_dir + 'grad_norm/')
        os.makedirs(overview_dir + 'loss/')

    loop_directories(result_folder,overview_dir) 
    print('Done')                   

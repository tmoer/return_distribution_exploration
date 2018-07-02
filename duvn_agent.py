# -*- coding: utf-8 -*-
"""
Double Uncertain Value Network implementation
@author: thomas
"""

# General imports
import tensorflow as tf
import numpy as np
import logging
import time
import random
import resource

# Custom imports
from rl.make_game import make_game, check_space
from rl.rltils import PartialRollout
from rl.policies import thompson_policy, egreedy_policy, ucb_policy
from rl.targets import calculate_targets, calculate_targets_off_policy

from network.duvn_nets import make_network
from network.graphtils import model_description
from utils.data import Database, Replay
from utils.distributed import make_init_ops
from utils.pytils import EMA, Interval_checker, TimedResults
from process import downsample

# Some settings
np.set_printoptions(threshold=np.nan,precision=3,suppress=True)
logger = logging.getLogger('root')
logger.propagate = False

class Agent(object):
    
    def __init__(self,hps,cluster_spec=None):
        self.hps = hps

        if hps.distributed:        
            ps_servers = ["/job:ps/task:{}".format(ps_num) for ps_num in range(hps.n_ps)]
            config = tf.ConfigProto(
                    device_filters=ps_servers + ['/job:ps', '/job:worker/task:{}'.format(hps.job_index)],
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
                    
        # Environment
        Env = make_game(hps.game)
        try:
            hps.max_ep_len = Env._max_episode_steps - 1
            logger.info('Set max steps per episode to {}'.format(hps.max_ep_len))
        except:
            logger.info('Environment does not have a time limit wrapper, using {} max steps per episode'.format(hps.max_ep_len))
            
        hps.action_dim, hps.action_discrete  = check_space(Env.action_space)
        hps.state_dim, hps.state_discrete  = check_space(Env.observation_space)       
        if not hps.action_discrete: raise ValueError('Continuous action space not implemented')

        # Seed        
        seed = np.random.randint(1e8) + 7*hps.job_index
        np.random.seed(seed)
        random.seed(seed)

        # Network  
        model, target_model, copy_op, global_model, global_target_model, global_copy_op, sync_op =  make_network(hps,cluster_spec)        
        logger.info('Total number of trainable parameters {} million'.format(model_description(model.var_list)/(1e6)))
        if not hps.distributed:
            with tf.Session() as sess:
                logger.info('Initializing ..')
                sess.run(tf.global_variables_initializer())
                run(Env,hps,sess,model,target_model,copy_op,global_model,global_target_model,global_copy_op, sync_op)
        else:
            # make init op
            global_init_op,local_init_op,global_variables = make_init_ops()
            sv = tf.train.Supervisor(is_chief=(hps.job_index == 0),
                             init_op=global_init_op,
                             local_init_op=local_init_op,
                             ready_op=tf.report_uninitialized_variables(tf.global_variables()),
                             #logdir = hps.result_dir,
                             )
            #print('Im worker {} before the supervisor'.format(hps.job_index))
            
            with sv.managed_session(server.target, config=config) as sess, sess.as_default():
                #print('Im worker {} after the supervisor'.format(hps.job_index))
                sess.run(sync_op)
                
                run(Env,hps,sess,model,target_model,copy_op,global_model,global_target_model,global_copy_op,sync_op,sv)
            
            sv.stop()
    
def run(Env,hps,sess,model,target_model=None,copy_op=None,global_model=None,global_target_model=None,global_copy_op=None,sync_op=None,sv=None):
    begin = overall_begin = time.time()
    print('Im worker {} and starting'.format(hps.job_index))

    # Database        
    D = Database(data_size=hps.n_ep_collect * hps.max_ep_len * hps.n_rep_target * 2,batch_size=hps.batch_size,entry_type='sequential')
    if hps.replay_size > 0:
        D_sars = Replay(max_size=hps.replay_size,prioritized_frac=hps.prioritized_frac)
        #D_sars = Database(data_size=hps.replay_size, batch_size=hps.batch_size,entry_type='sequential')       
        
    #saver = tf.train.Saver(max_to_keep=10)        

    # Counters
    Copy_count = Interval_checker(hps.max_ep,hps.t_max,hps.store_copy_freq)
    Eval_count = Interval_checker(hps.max_ep,hps.t_max,hps.evaluate_freq)
    if hps.game == 'Toy' and hps.visualize:
        from rl.envs.toy import ToyPlotter, ToyDomainPlotter
        toy_plotter = ToyPlotter()
        toy_domain_plotter = ToyDomainPlotter(hps)
    elif 'Chain' in hps.game and hps.visualize:
        from rl.envs.chain import ChainPlotter, ChainDomainPlotter
        chain_plotter = ChainPlotter(Env.correct,n_plot=Env.n)
        chain_domain_plotter = ChainDomainPlotter(Env)
    # Discretizer = TransformDiscrete(hps.disc_n,hps.disc_min,hps.disc_max)
    # ema = EMA()    
    if 'Chain' in hps.game:
        # necessary for plotting visitation counts correctly
        test_Env = make_game(hps.game)
        test_Env.correct = Env.correct
    else:
        test_Env = Env
        
    t = 0     
    ep = 0
    log_count = 0
    time_result = TimedResults(n=6) # Average Reward, Empirical Loss, Qsa_norm, grad_norm, loss
    #epsilon = AnnealLinear(hps.e_init,hps.e_final,int(hps.anneal_frac*hps.t_max))                
        
    time_check = 't < hps.t_max' if not hps.distributed else '(t < hps.t_max) and not sv.should_stop()'
    running_mean = 0.0
    frac = 0.97
    
    while eval(time_check): # train loop
        now = time.time()
        if hps.distributed:
            sess.run(sync_op)            
        e = sess.run(model.epsilon)

        # Collect data
        t_new,t_,av_R,ep_R,data = collect_data(hps,model,sess,Env,e)  
        
        # Process new (on-policy) data
        D.clear() # clear training database
        D,D_sars,Qsa_norm,Qsa_sds = calculate_targets(data,D,D_sars,model,sess,hps,hps.lambda_,off_policy=hps.off_policy,target_model=target_model)

        # Fill up database from replay
        if D_sars.size > hps.min_replay_size:
            extra_needed = hps.replay_frac * D.size
            more = hps.batch_size - (D.size + extra_needed) % hps.batch_size
            if more > 0:
                extra_needed += more  # to make full minibatches from rollout
            if (extra_needed > 0) and (extra_needed <= D_sars.size):
                # draw the samples
                sb,ab,rb,s1b,tb = D_sars.sample_random_batch(extra_needed,True) # sample from replay
                D,D_sars,Qsa_norm2 = calculate_targets_off_policy(sb,ab,rb,s1b,tb,D,D_sars,model,sess,hps,target_model) # Process the extra data (off-policy)

        # Train
        gradient_norm,clipped_norm,loss = train(hps,model,sess,D)

#        # Put new data in replay database
#        if hps.replay_size > 0:
#            for rollout in data:
#                si,ai,ri,st = rollout.extract()
#                tt = np.zeros(si.shape[0])
#                tt[-1] = rollout.terminal
#                si1 = np.concatenate([si[1:,],st[None,:]],axis=0)
#                D_sars.store_from_array(si,ai,ri,si1,tt)              


#        # Send data to replay database    
#        if hps.replay_frac > 0:
#            if hps.replay_size > 0:
#                for j,rollout_data in enumerate(data):
#                    rollout_data.seed = None # seed becomes irrelevant
#                    D_replay.put(rollout_data,priority=prios)#-1.0*prios[j])
#    
#            if hps.replay_size > 0 and len(D_replay.q)>(hps.replay_frac*len(data)*2):
#                replay_data = D_replay.get(hps.replay_frac*len(data))
#                # replay always off-policy (lambda_=0.0) for sample-based loss
#                D,_ = calculate_targets(replay_data,D,model,sess,hps,lambda_=0.0,off_policy=hps.off_policy,target_model=target_model)
#                new_prios = 0.0
#                #D,_,new_prios = process(replay_data,D,model,sess,hps,target_model=target_model,ema=ema)
#                train(hps,model,sess,D)
#                D.clear()             
#                if hps.prioritized_frac>0.0:
#                    for i in range(len(replay_data)):
#                        D_replay.put(replay_data[i],priority=-1.0*new_prios[i])
                                                                                 
                    
        # Counters        
        _,_,t,ep = sess.run([model.inc_t,model.inc_ep,model.global_t,model.global_ep],feed_dict={model.t:t_new,model.ep:len(ep_R)})

        if hps.level == 'debug':
            memory_use = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            print('Process {} = worker {} loops in {} seconds, memory use {} Mb for replay size {}'.format(hps.agent_index,hps.job_index,time.time() - now,memory_use,D_sars.size))
            #else:
            #    print('Memory use {} Mb'.format(memory_use))

        if (not hps.distributed) or (hps.distributed and hps.job_index == 0):
            # Store
            episode_reward,average_reward = np.mean(ep_R),np.mean(av_R)  
            time_result.add([ep - (len(ep_R)/2)],[average_reward],[episode_reward],[np.mean(Qsa_norm)],[gradient_norm],[loss],[int(t-t_new/2)])                
            #logger.info('Evaluated: Ep {:5d}, t {:5d}: Ep return {:4.2f}, Running mean {:4.2f}, Qsa_norm {:4.2f}, grad_norm {:4.2f}, clipped_norm {:4.2f}, loss {:4.2f},  episode_length {:3.1f}'.format(ep,t,episode_reward,running_mean,np.mean(Qsa_norm),gradient_norm,clipped_norm,loss,np.mean(t_)))
            ep_curve,av_R_curve,ep_R_curve,Qsa_norm_curve,grad_norm_curve,loss_curve,t_curve = time_result.extract()                
            downsample_store(hps,ep_curve,ep_R_curve,av_R_curve,Qsa_norm_curve,grad_norm_curve,loss_curve,t_curve,out_len=1000)            

            # Logging
            if (t // hps.log_interval > log_count) and (not hps.slurm):
                running_mean = frac*running_mean + (1-frac)*np.mean(ep_R)
                logger.info('Ep {:4d}, t {:5d}: Ep Return {:4.2f}, Run_mean {:4.2f}, Qsa_mean {:4.2f}, Qsa_sd {:4.2f}, grad_norm {:4.2f}, clip_norm {:4.2f}, loss {:4.2f}, ep_len {:3.1f}'.format(ep,t,np.mean(ep_R),running_mean,np.mean(Qsa_norm),Qsa_sds,gradient_norm,clipped_norm,loss,np.mean(t_)))
                log_count += 1

            # Copy target network
            if Copy_count.should_update(t,ep):
                # Target net
                if hps.target_net:
                    if hps.distributed:
                        sess.run(global_copy_op)
                    else:
                        sess.run(copy_op)
           
            # for uncer == 'log_bay' --> set new uncertainty estimates
            # sample larger X and Y batch:
            if hps.uncer == 'log_bay':
                D.clear()
                log_bay_sample = np.min([D_sars.size,3000])
                sb,ab,rb,s1b,tb = D_sars.sample_random_batch(log_bay_sample,True) # sample from replay
                D,D_sars,Qsa_norm2 = calculate_targets_off_policy(sb,ab,rb,s1b,tb,D,D_sars,model,sess,hps,target_model) # Process the extra data (off-policy)
                s_batch,a_batch,y_batch = D.sample_random_batch(log_bay_sample)
                seed = [np.random.randint(1e15),np.random.randint(1e15)] # just need to feed something new for each batch
                feed_dict = {model.x: s_batch,
                     model.a: a_batch,
                     model.y: y_batch,
                     model.seed:seed}               
                sess.run(model.lin_bay_update, feed_dict = feed_dict)           

            # Visualize for Toy/Chain
            if hps.game == 'Toy' and hps.visualize:
                if Eval_count.should_update(t,ep):
                    toy_plotter.update(sess,model,hps,ep) 
                    toy_domain_plotter.update(Env.counts)
            elif 'Chain' in hps.game and hps.visualize:
                if Eval_count.should_update(t,ep):
                    chain_plotter.update(sess,model,hps,ep)
                    chain_domain_plotter.update(Env.counts)
            elif hps.visualize:
                if Eval_count.should_update(t,ep):                                
                    episode_reward,average_reward = evaluate(test_Env,hps,model,sess)                
                
            # Store model
            #if rep == 0:
                # saver.save(sess,hps.result_dir+make_name('',hps.item1,item1,hps.item2,item2,hps.item3,item3,hps.item4,item4)+'model.ckpt')

#            if Eval_count.should_update(t,ep):
#                if hps.distributed:
#                    sess.run(sync_op)
#                episode_reward,average_reward = evaluate(test_Env,hps,model,sess)                
#                time_result.add([ep - (len(ep_R)/2)],[average_reward],[episode_reward],[np.mean(ep_R)],[np.mean(Qsa_norm)],[gradient_norm],[loss])                
#                logger.info('Evaluated: Ep {:5d}, t {:5d}: Ep return {:4.2f}, Running mean {:4.2f}, Qsa_norm {:4.2f}, grad_norm {:4.2f}, clipped_norm {:4.2f}, loss {:4.2f},  episode_length {:3.1f}'.format(ep,t,episode_reward,running_mean,np.mean(Qsa_norm),gradient_norm,clipped_norm,loss,np.mean(t_)))
#                ep_curve,av_R_curve,ep_R_curve,Qsa_norm_curve,grad_norm_curve,loss_curve = time_result.extract()                
#                downsample_store(hps,ep_curve,ep_R_curve,av_R_curve,Qsa_norm_curve,grad_norm_curve,loss_curve,out_len=1000)            

        if (ep > hps.max_ep) or (t > hps.t_max):
            if hps.distributed:
                sv.request_stop()
            break # max number of episodes overrules max number of steps
        elapsed = (time.time()-overall_begin)/60
        logger.info('Reached {} episodes in {} timesteps, took {} hours = {} minutes'.format(ep,t,elapsed/60,elapsed))

    
    if (not hps.distributed) or (hps.disributed and hps.job_index == 0):
        ep_curve,av_R_curve,ep_R_curve,Qsa_norm_curve,grad_norm_curve,loss_curve,t_curve = time_result.extract()                
        downsample_store(hps,ep_curve,ep_R_curve,av_R_curve,Qsa_norm_curve,grad_norm_curve,loss_curve,t_curve,out_len=1000)            
    
        # Compute best result (last 10%)
        fraction = 0.1
        save_from_index = int(np.ceil(len(ep_curve)*fraction))
        ep_R_best = np.mean(ep_R_curve[save_from_index:])                        
        av_R_best = np.mean(av_R_curve[save_from_index:]) 
        np.savetxt(hps.result_dir+'best_results.txt',np.append(ep_R_best,av_R_best),fmt='%.3g') 


def evaluate(Env,hps,model,sess):
    R_ep = []
    R_av = []
    for i in range(hps.n_eval):
        s = Env.reset()
        s = correct_dim(s)
        R = 0
        seed = [np.random.randint(1e15),np.random.randint(1e15)] if hps.fix_seed_per_roll_out else None
        for j in range(hps.max_ep_len):
            local_seed = seed if seed is not None else [np.random.randint(1e15),np.random.randint(1e15)] # sample the local seed
            if hps.policy == 'thompson':
                a = thompson_policy(s[None,:],model,sess,hps,local_seed,hps.eval_on_mean_output,hps.eval_on_mean_params)
            elif hps.policy == 'egreedy':
                a = egreedy_policy(s[None,:],model,sess,hps,e=0.0,seed=local_seed)
            elif hps.policy == 'ucb':
                a = ucb_policy(s[None,:],model,sess,hps,local_seed)#,hps.eval_on_mean_output,hps.eval_on_mean_params)
            a = a[0]
            #print(s,a)
            #print(hps.eval_on_mean_params,hps.eval_on_mean_output)
            
            s1,r,terminal,info = Env.step(correct_action_dim(a,model.action_dim))
            if hps.visualize:
                Env.render()
            s1 = correct_dim(s1)
            s = s1
            R += r
            if terminal:
                break
        R_ep.append(R)
        R_av.append(R/(j+1))
    return np.mean(R_ep),np.mean(R_av)
                                                                      
def downsample_store(hps,ep_curve,ep_R_curve,av_R_curve,Qsa_norm_curve,grad_norm_curve,loss_curve,t_curve,out_len=1000):
    # Downsample
    ep_curve = downsample(ep_curve,out_len)
    ep_R_curve = downsample(ep_R_curve,out_len)
    av_R_curve = downsample(av_R_curve,out_len)
    Qsa_norm_curve = downsample(Qsa_norm_curve,out_len)
    grad_norm_curve = downsample(grad_norm_curve,out_len)
    loss_curve = downsample(loss_curve,out_len)
    t_curve = downsample(t_curve,out_len)
    
    # Write results to files    
    np.savetxt(hps.result_dir+'episode_raw.txt',ep_curve,fmt='%.3g')                    
    np.savetxt(hps.result_dir+'av_reward_raw.txt',av_R_curve,fmt='%.3g')
    np.savetxt(hps.result_dir+'ep_reward_raw.txt',ep_R_curve,fmt='%.3g') 
    np.savetxt(hps.result_dir+'Qsa_norm_raw.txt',Qsa_norm_curve,fmt='%.3g') 
    np.savetxt(hps.result_dir+'grad_norm_raw.txt',grad_norm_curve,fmt='%.3g') 
    np.savetxt(hps.result_dir+'loss_raw.txt',loss_curve,fmt='%.3g') 
    np.savetxt(hps.result_dir+'t_raw.txt',t_curve,fmt='%.3g') 

def train(hps,model,sess,D):
    gradient_norms = []
    clipped_norms = []
    losses = []
    n_epochs = hps.n_epochs
    lr = hps.lr
    if hps.uncer == 'vi':
        n_epochs *= hps.k
        lr /= hps.k
    for l in range(n_epochs):
        for batch in D:
            s_batch,a_batch,y_batch = batch
            batch_size = s_batch.shape[0]
            #print(np.concatenate([s_batch,a_batch,y_batch],axis=1))
            seed = [np.random.randint(1e15),np.random.randint(1e15)] # just need to feed something new for each batch
            feed_dict = {model.x: s_batch,
                         model.a: a_batch,
                         model.y: y_batch,
                         model.seed:seed,
                         #model.lr : lr,
                         model.batch_size : batch_size}
            _,emp_loss,gradient_norm,clipped_norm = sess.run([model.train_op,model.loss,model.gradient_norm,model.clipped_norm],feed_dict = feed_dict)
            gradient_norms.append(gradient_norm)
            clipped_norms.append(clipped_norm)
            losses.append(emp_loss)
    return np.mean(gradient_norms),np.mean(clipped_norms),np.mean(losses)

def correct_dim(x):
    if type(x) == int or type(x) == np.int64: 
        x = np.array([x])  
    return x

def correct_action_dim(a,action_discrete=True):
    ''' removes a dimension for discrete action space, as OpenAI Gym expects an integer '''
    if action_discrete:
        a = a[0]
    return a

def collect_data(hps,model,sess,Env,e):
    ''' Collects data '''
    ep = 0 # episode counter
    t = 0 # timestep counter
    t_,R_mean,R_sum,data = [],[],[],[] # data per episode
    
    while ep < hps.n_ep_collect:
        terminal = False
        s = Env.reset()
        s = correct_dim(s)
        seed = [np.random.randint(1e15),np.random.randint(1e15)] if hps.fix_seed_per_roll_out else None
        rollout_data = PartialRollout()
        if seed is not None: 
            rollout_data.seed = seed
        terminal = False
            
        while (not terminal) and  (rollout_data.t < hps.max_ep_len):
            local_seed = seed if seed is not None else [np.random.randint(1e15),np.random.randint(1e15)] # sample the local seed
            # Policy
            try:
                if hps.policy == 'thompson':
                    a = thompson_policy(s[None,:],model,sess,hps,seed=local_seed)
                elif hps.policy == 'egreedy':
                    e = sess.run(model.epsilon)
                    a = egreedy_policy(s[None,:],model,sess,hps,e=e,seed=local_seed)
                elif hps.policy == 'ucb':
                    a = ucb_policy(s[None,:],model,sess,hps,seed=local_seed)
                a = a[0]
            except Exception as e:
                print('hps.policy = {}, s = {} exception = {}'.format(hps.policy,s,e))
    
            # Steps
            s1,r,terminal,info = Env.step(correct_action_dim(a,model.action_discrete))
            s1 = correct_dim(s1)
            rollout_data.add(s,a,[r])   
            s = s1
        rollout_data.add_last_state(s,terminal) # add the last state
        
        data.append(rollout_data)        
        ep += 1
        t += rollout_data.t

        R_sum.append(rollout_data.r_sum)
        R_mean.append(rollout_data.r_sum/rollout_data.t)
        t_.append(rollout_data.t)

        #logger.debug('Episode reward {}'.format(rollout_data.r_sum))        
    return t,t_,R_mean,R_sum,data

def roll_out(hps,Env,model,sess,s,e,seed):
    ''' Performs a simple rollout '''
    rollout_data = PartialRollout()
    if seed is not None: rollout_data.seed = seed
    while (not rollout_data.terminal) and  (rollout_data.t < hps.max_roll_len):
        local_seed = seed if seed is not None else [np.random.randint(1e15),np.random.randint(1e15)] # sample the local seed
        # Policy
        if hps.policy == 'thompson':
            a = thompson_policy(s[None,:],model,sess,hps,seed=local_seed)
        elif hps.policy == 'egreedy':
            e = sess.run(model.epsilon)
            a = egreedy_policy(s[None,:],model,sess,hps,e=e,seed=local_seed)
        elif hps.policy == 'ucb':
            a = ucb_policy(s[None,:],model,sess,hps,seed=local_seed)
        a = a[0]

        # Steps
        s1,r,terminal,info = Env.step(correct_action_dim(a,model.action_discrete))
        s1 = correct_dim(s1)
        rollout_data.add(s,a,[r],terminal)   
        s = s1
    rollout_data.add_last_state(s) # add the last state
    return rollout_data,s

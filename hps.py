# -*- coding: utf-8 -*-
"""
Default hyperparameter settings
@author: thomas
"""
 
def get_hps():
    ''' Hyperparameter settings '''
    return HParams(      
        # General
        game = 'Chain-15', # Environment name
        name = 'None', # Name of experiment
        t_max = 2500, # max timesteps - terminates on min of t_max or max_ep
        max_ep = 2500, # max episodes

        # Slurm parameters
        slurm = False,
        slurm_time = '3:59:59',
        slurm_mem = 2048,
        mem_per_cpu = 2048,
        n_cpu = 2,
        cpu_per_task = 2, 
        slurm_qos = 'short',        

        # Distributed training
        distributed = False,
        n_ps = 2,
        n_agent = 4,
        agent_index = 0,
        job_type = 'worker',
        job_index = 0,
        
        # Directories
        base_result_dir = '',
        result_dir = '',
        checkpoint_dir = '',
        rep = 0, # repetition index
        log_interval = 50,

        # Network
        network = 'multiple', # 'single' or 'multiple' (network heads)
        n_hidden = 256, # neurons in hidden part
        n_layers = 2, # number of hidden layers (in fully connected part)
        nonlin = 'elu',
        
        # output distribution loss        
        output = 'gaussian', # deterministic, gaussian, categorical, mog
        loss = 'analytic', # analytic, sample
        distance = 'ce', # kl,bhat, hel --> type of distance for gaussian output distribution
        sd_output_bias = 1.0,
        n_bins = 51, # number of bins for output = categorical
        cat_min = -1.0, # minimum bin bound for output = categorical
        cat_max = 1.0, # maximum bin bound for output = categorical
        n_mix = 5, # number of mixture for output = mog

        # Parameter uncertainty
        uncer = 'dropout', # dropout, vi
        p_dropout = 1.0, # probability to keep a node
        prior_type = 'spike', # 'spike' for spike-and-slab, 'gaussian'
        sigma_prior = 1.0, # sigma of normal prior for variational inference
        kl_mode = 'sample', # kl mode for vi on model parameters
        k = 4, # number of samples per datapoint for uncer = 'vi'

        n_rep_mc = 2, # number of monte carlo integration samples over the next state distribution
        n_rep_target = 1, # number of repeated draws for 'sample' based loss. Note: nested draws within n_rep_mc
        
        # Policy
        policy = 'thompson', # 'thompson', 'egreedy', 'ucb'
        thompson_epsilon = 0.0, # potential small epsilon always applied
        fix_seed_per_roll_out = False, # fixes the parametric uncertainty seed over an entire rollout
        fix_seed_for_target = False,
        n_thompson_sample = 1, # number of draws before maxing within thompson sampling. Higher values move Thompson sampling towards a ucb policy. 
        
        # Replay & Target net
        replay_size = 100000,
        min_replay_size = 5000,
        replay_frac = 1, # number of times an episode gets replayed
        prioritized_frac = 0.0,
        priority_type = 'loss', # 'loss' (distributional) or 'td' (means)
        target_net = True,     
        store_copy_freq = 700, # total number of times the target net will be copied
        evaluate_freq = 750, # total number of evalutions
        
        # database
        n_ep_collect = 1,
        max_ep_len = 200,
        max_roll_len = 32,
        batch_size = 32,
        
        # Training
        n_epochs = 1,
        lr = 0.001,
        decay_lr = 'None',
        decay_rate = 0.01,
        clip_gradients = False,
        clip_value = 1.0, # max gradient per parameter
        clip_global = 0.0, # max global gradient 
        
        # RL basics
        gamma = 0.995,
        lambda_= 0.0, # in GAE
        off_policy = False, # Note: replay is always off-policy, this concerns the initial online processing of a roll-out     
        off_policy_on_mean = False,        
        double_dqn = False,
     
        # for e-greedy & boltzmann
        e_init = 0.05,
        e_final= 0.05,
        anneal_frac = 0.1,

        # Old - unused
        modify_pilco = False,
        normalize_Q = False,
        offset = 0.0, # 
        
        # Evaluation
        n_eval = 2, # evaluation episodes
        eval_on_mean_output = False,
        eval_on_mean_params = False,
        visualize = True, # only for 'Toy' environment
        n_rep_visualize = 3, # number of MC samples over the parametric uncertainty in the visualization
        
        # Hyperparameter looping
        n_rep = 1,
        loop_hyper = False,
        item1 = None,
        seq1 = [None],
        item2 = None,
        seq2 = [None],
        item3 = None,
        seq3 = [None],
        item4 = None,
        seq4 = [None],

        level='info',
        )
        
class HParams(object):

    def __init__(self, **kwargs):
        self._items = {}
        for k, v in kwargs.items():
            self._set(k, v)

    def _set(self, k, v):
        self._items[k] = v
        setattr(self, k, v)
        
    def _get(self,k):
        return self._items[k]
        
    def __eq__(self, other) : 
        return self.__dict__ == other.__dict__

    def parse(self, str_value):
        hps = HParams(**self._items)
        for entry in str_value.strip().split(","):
            entry = entry.strip()
            if not entry:
                continue
            key, sep, value = entry.partition("=")
            if not sep:
                raise ValueError("Unable to parse: %s" % entry)
            default_value = hps._items[key]
            if isinstance(default_value, bool):
                hps._set(key, value.lower() == "true")
            elif isinstance(default_value, int):
                hps._set(key, int(value))
            elif default_value is None and value == 'None':
                hps._set(key, None)
            elif isinstance(default_value, float):
                hps._set(key, float(value))
            elif isinstance(default_value, list):
                value = value.split('+')
                default_inlist = hps._items[key][0]
                if key == 'seq1':
                    default_inlist = hps._items[hps._items['item1']]
                if key == 'seq2':
                   default_inlist = hps._items[hps._items['item2']]
                if key == 'seq3':
                    default_inlist = hps._items[hps._items['item3']]
                if key == 'seq4':
                   default_inlist = hps._items[hps._items['item4']]
                if isinstance(default_inlist, bool):
                    hps._set(key, [i.lower() == "true" for i in value])
                elif isinstance(default_inlist, int):
                    hps._set(key, [int(i) for i in value])
                elif isinstance(default_inlist, float):
                    hps._set(key, [float(i) for i in value])
                else:
                    hps._set(key,value) # string
            else:
                hps._set(key, value)
        return hps

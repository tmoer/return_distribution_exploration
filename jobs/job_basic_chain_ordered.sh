#game=ChainOrdered-10 and max_ep=3000
python3 setup.py --hpconfig name=combined,loop_hyper=True,item1=policy,seq1=ucb+thompson-1+thompson-3+egreedy,item2=output,seq2=deterministic+gaussian+categorical+mog,item3=uncer,seq3=dropout-1.0+vi-1.0+vi-0.1,item4=lr,seq4=0.005+0.0005,store_copy_freq=500,target_net=False,replay_frac=3,n_epochs=4,n_ep_collect=4,loss=analytic,n_rep_mc=1,n_rep=3,game=ChainOrdered-10,max_ep=3000,t_max=10000000,slurm_qos=short,slurm_time=0-03:59:00,slurm=True,level=error,n_bins=5,cat_min=0.0,cat_max=1.0,sd_output_bias=1.0,distributed=False,n_ps=2,n_agent=16

#game=ChainOrdered-30 and max_ep=3000
python3 setup.py --hpconfig name=combined,loop_hyper=True,item1=policy,seq1=ucb+thompson-1+thompson-3+egreedy,item2=output,seq2=deterministic+gaussian+categorical+mog,item3=uncer,seq3=dropout-1.0+vi-1.0+vi-0.1,item4=lr,seq4=0.005+0.0005,store_copy_freq=500,target_net=False,replay_frac=3,n_epochs=4,n_ep_collect=4,loss=analytic,n_rep_mc=1,n_rep=3,game=ChainOrdered-30,max_ep=3000,t_max=10000000,slurm_qos=short,slurm_time=0-03:59:00,slurm=True,level=error,n_bins=5,cat_min=0.0,cat_max=1.0,sd_output_bias=1.0,distributed=False,n_ps=2,n_agent=16

#game=ChainOrdered-50 and max_ep=3000
python3 setup.py --hpconfig name=combined,loop_hyper=True,item1=policy,seq1=ucb+thompson-1+thompson-3+egreedy,item2=output,seq2=deterministic+gaussian+categorical+mog,item3=uncer,seq3=dropout-1.0+vi-1.0+vi-0.1,item4=lr,seq4=0.005+0.0005,store_copy_freq=500,target_net=False,replay_frac=3,n_epochs=4,n_ep_collect=4,loss=analytic,n_rep_mc=1,n_rep=3,game=ChainOrdered-50,max_ep=3000,t_max=10000000,slurm_qos=short,slurm_time=0-03:59:00,slurm=True,level=error,n_bins=5,cat_min=0.0,cat_max=1.0,sd_output_bias=1.0,distributed=False,n_ps=2,n_agent=16

#game=ChainOrdered-100 and max_ep=3000
python3 setup.py --hpconfig name=combined,loop_hyper=True,item1=policy,seq1=ucb+thompson-1+thompson-3+egreedy,item2=output,seq2=deterministic+gaussian+categorical+mog,item3=uncer,seq3=dropout-1.0+vi-1.0+vi-0.1,item4=lr,seq4=0.005+0.0005,store_copy_freq=500,target_net=False,replay_frac=3,n_epochs=4,n_ep_collect=4,loss=analytic,n_rep_mc=1,n_rep=3,game=ChainOrdered-100,max_ep=3000,t_max=10000000,slurm_qos=short,slurm_time=0-03:59:00,slurm=True,level=error,n_bins=5,cat_min=0.0,cat_max=1.0,sd_output_bias=1.0,distributed=False,n_ps=2,n_agent=16


# First combined experiment: policy, output, dropout, lr
# python3 setup.py --hpconfig name=combined,loop_hyper=True,item1=policy,seq1=ucb+thompson+egreedy,item2=output,seq2=deterministic+gaussian+categorical+mog,item3=p_dropout,seq3=1.0+0.85+0.7,item4=lr,seq4=0.005+0.0005,evaluate_freq=1500,target_net=True,replay_frac=2,n_ep_collect=3,loss=analytic,n_rep_mc=5,n_thompson_sample=5,n_rep=2,store_copy_freq=1000

# More runs
python3 setup.py --hpconfig name=combined,loop_hyper=True,item1=policy,seq1=ucb+thompson-1+thompson-3+egreedy,item2=output,seq2=deterministic+gaussian+categorical+mog,item3=uncer,seq3=dropout-1.0+vi-1.0+vi-0.1,item4=lr,seq4=0.005+0.0005,store_copy_freq=750,target_net=False,replay_frac=1,n_epochs=1,n_ep_collect=3,loss=analytic,n_rep_mc=2,n_rep=10


#python3 setup.py --hpconfig name=combined,loop_hyper=True,item1=policy,seq1=ucb+thompson-1+egreedy,item2=output,seq2=deterministic+gaussian+categorical,item3=uncer,seq3=dropout-1.0+dropout-0.75+vi-0.4,item4=lr,seq4=0.005+0.0005,evaluate_freq=1500,target_net=False,replay_frac=1,n_ep_collect=4,loss=analytic,n_rep_mc=2,n_rep=1





#python3 setup.py --hpconfig name=combined,loop_hyper=True,item1=policy,seq1=ucb+thompson+egreedy,item2=output,seq2=deterministic+gaussian+categorical+mog,item3=p_dropout,seq3=1.0+0.85+0.7,item4=lr,seq4=0.001+0.0001,evaluate_freq=1500,target_net=True,replay_frac=2,n_ep_collect=3,loss=analytic,n_rep_mc=4,n_thompson_sample=5,n_rep=5,store_copy_freq=1000,loss=sample

#python3 setup.py --hpconfig name=thompson_sample,loop_hyper=True,item1=n_thompson_sample,seq1=1+2+5+10,item2=output,seq2=deterministic+gaussian+categorical+mog,item3=p_dropout,seq3=1.0+0.85+0.7,item4=lr,seq4=0.001+0.0001,evaluate_freq=1500,target_net=True,replay_frac=2,n_ep_collect=3,loss=analytic,n_rep_mc=4,n_rep=2,store_copy_freq=1000,loss=sample,policy=thompson



# More runs
#python3 setup.py --hpconfig name=combined,loop_hyper=True,item1=policy,seq1=ucb+thompson+egreedy,item2=output,seq2=deterministic+gaussian+categorical+mog,item3=p_dropout,seq3=1.0+0.85+0.7,item4=lr,seq4=0.001+0.0001,evaluate_freq=1500,target_net=False,replay_frac=2,n_ep_collect=3,loss=analytic,n_rep_mc=4,n_thompson_sample=1,n_rep=5,store_copy_freq=1000,loss=analytic

#python3 setup.py --hpconfig name=combined,loop_hyper=True,item1=policy,seq1=ucb+thompson+egreedy,item2=output,seq2=deterministic+gaussian+categorical+mog,item3=p_dropout,seq3=1.0+0.85+0.7,item4=lr,seq4=0.001+0.0001,evaluate_freq=1500,target_net=True,replay_frac=2,n_ep_collect=3,loss=analytic,n_rep_mc=4,n_thompson_sample=5,n_rep=5,store_copy_freq=1000,loss=sample

#python3 setup.py --hpconfig name=thompson_sample,loop_hyper=True,item1=n_thompson_sample,seq1=1+2+5+10,item2=output,seq2=deterministic+gaussian+categorical+mog,item3=p_dropout,seq3=1.0+0.85+0.7,item4=lr,seq4=0.001+0.0001,evaluate_freq=1500,target_net=True,replay_frac=2,n_ep_collect=3,loss=analytic,n_rep_mc=4,n_rep=2,store_copy_freq=1000,loss=sample,policy=thompson



#python3 setup.py --hpconfig #name=categorical,n_rep=3,output=categorical,policy=ucb,output=categorical,loop_hyper=True,item1=p_dropout,seq1=1.0+0.85+0.7
#python3 setup.py --hpconfig name=gaussian,n_rep=3,policy=ucb,output=gaussian,loop_hyper=True,item1=p_dropout,seq1=1.0+0.85+0.7
#python3 setup.py --hpconfig name=deterministic,n_rep=3,policy=ucb,output=deterministic,loop_hyper=True,item1=p_dropout,seq1=1.0+0.85+0.7
#python3 setup.py --hpconfig name=thompson,n_rep=3,policy=thompson,output=categorical,loop_hyper=True,item1=p_dropout,seq1=1.0+0.85+0.7,item2=n_thompson_sample,seq2=1+5+20
#python3 setup.py --hpconfig name=egreedy,n_rep=3,policy=egreedy,output=deterministic,p_dropout=1.0,e_init=0.05,e_final=0.05,anneal_frac=0.01


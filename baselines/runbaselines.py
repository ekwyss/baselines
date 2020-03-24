from run import main

main(['/home/eric/baselines/baselines/run.py', '--alg=her', '--env=FetchPickAndPlace-v1', '--num_timesteps=100000', '--num_env=2', '--num_cpu=1', '--demo_file=/home/eric/baselines/demonstration_data/data_fetch_random_100goalind_5_5_20.npz', '--save_path=/home/eric/baselines/delete.pkl', '--bc_loss=1', '--q_filter=1', '--num_demo=100', '--demo_batch_size=128', '--prm_loss_weight=0.001', '--aux_loss_weight=0.0078', '--n_cycles=20', '--batch_size=1024', '--random_eps=0.1', '--noise_eps=0.1'])

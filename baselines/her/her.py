import os

import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds, tf_util
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker

from baselines.her.policies import Policies
import pickle

def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]

# def split_episode(episode):
#     #currently one episode case
#     new_episodes = []
#     goal_indices = np.where(episode['info_is_success'][0] == 1)[0] + 1
#     # print(goal_indices)
#     #Change nothing, just put all into first (two) policy(s)
#     if len(goal_indices) == 0:
#         return [episode]
#     #trailing successes irrelevant, will just pass everything to end
#     if len(goal_indices) > 2:
#         goal_indices = goal_indices[:2]
#     #whether 1,2,... always want last episode to reach to end
#     # goal_indices.append(-1)
#     goal_indices = np.concatenate(([0], goal_indices, [-1]))
#     # print(goal_indices)
#     for i in range(len(goal_indices)-1):
#         # new_dict = dict.fromkeys(episode.keys())
#         new_dict = {}
#         for key in episode.keys():
#             # temp = []
#             fst_ind = goal_indices[i]
#             snd_ind = goal_indices[i+1]
#             if (key == 'o' or key == 'ag') and snd_ind != -1:
#                 snd_ind += 1 
#             #for multi episode case - doesn't work
#             # for ep in episode[key]:
#                 ## new_dict[key][0] = episode[key][0]
#                 # temp.append(ep[fst_ind:snd_ind])
#             # new_dict[key] = temp

#             new_dict[key] = np.array([episode[key][0][fst_ind:snd_ind]])
#         new_episodes.append(new_dict)
#     # print('hilo')
#     # print(new_episodes[0]['g'].shape)
#     # print(episode['g'].shape)
#     return new_episodes

def train(*, policies, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, demo_file, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')

    logger.info("Training...")
    best_success_rate = -1

    # if policy.bc_loss == 1: policy.init_demo_buffer(demo_file) #initialize demo buffer if training with demonstrations
    if policies.policies[0].bc_loss == 1: policies.init_demo_buffer(demo_file) #initialize demo buffer if training with demonstrations

    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    for epoch in range(n_epochs):
        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            # new_episodes = split_episode(episode)

            #ATTEMPTING TO ONLY UPDATE POLICIES USED IN EPISODE
            #don't think you can do this because there may be multiple episodes in the buffer
            # goal_indices = np.where(episode['info_is_success'][0] == 1)[0]
            #temp fix
            # new_episodes = [episode.copy(),episode.copy(),episode.copy()]

            # split up episodes into subsegments
            # policies.store_episodes(new_episodes)
            policies.store_episodes(episode)
            # policy.store_episode(episode)
            for _ in range(n_batches):
                policies.train()
                # policy.train()
            policies.update_target_nets()
            # policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        policies.record_logs(logger)
        # for key, val in policy.logs():
            # logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_path:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policies(best_policy_path)
            # evaluator.save_policy(best_policy_path)
            evaluator.save_policies(latest_policy_path)
            # evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policies(policy_path)
            # evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return policies


def learn(*, network, env, total_timesteps,
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=5,
    clip_return=True,
    demo_file=None,
    override_params=None,
    load_path=None,
    save_path=None,
    **kwargs
):

    override_params = override_params or {}
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    env_name = env.spec.id
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
         json.dump(params, f)
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs

    if demo_file is not None:
        params['bc_loss'] = 1
    params.update(kwargs)

    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    #if we had subgoals with different dims we would have to change that here/in configure_dims
    # then when initializing each ddpg agent we pass in the appropriate dims
    # print(params.keys())
    dims = config.configure_dims(params)
    # policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    policies = Policies(1,dims,params,clip_return, num_goals=3)
    # policies = Policies(2,dims,params,clip_return,3,[0,2])
    # policies = Policies(3,dims,params,clip_return)
    #same obs and action space, hack
    # policies = [config.configure_ddpg(dims=dims, params=params, clip_return=clip_return),config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)]
    if load_path is not None:
        tf_util.load_variables(load_path)

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = eval_env or env

    rollout_worker = RolloutWorker(env, policies, dims, logger, monitor=True, **rollout_params)
    # rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policies, dims, logger, **eval_params)
    # evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)

    n_cycles = params['n_cycles']
    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_batch_size

    return train(
        save_path=save_path, policies=policies, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file)


@click.command()
@click.option('--env', type=str, default='FetchReach-v1', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--total_timesteps', type=int, default=int(5e5), help='the number of timesteps to run')
@click.option('--seed', type=int, default=0, help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='future', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--demo_file', type=str, default = 'PATH/TO/DEMO/DATA/FILE.npz', help='demo data file path')
def main(**kwargs):
    learn(**kwargs)


if __name__ == '__main__':
    main()

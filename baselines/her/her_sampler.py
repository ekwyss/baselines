import numpy as np
import pickle

import random

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):#, policy_index):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):#, ep_Ts):

        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        
        #Matrix holding timesteps of transitions attempting to achieve each subgoal within each episode
        Ts_IJ = []
        for i in range(rollout_batch_size):
            Ts_i = []
            #TODO: Pass this information in somewhere (likely in make_sample_her_transitions like where policy_index was used for multiple_policy case)
            for j in range(3):
                Ts_ij = np.where(episode_batch['info_goal_index'][i] == j)[0]
                Ts_i.append(Ts_ij if len(Ts_ij) > 1 else np.array([],dtype=int))
                # Ts_i.append(np.where(episode_batch['info_goal_index'][i] == j)[0])
            Ts_IJ.append(np.array(Ts_i))
        Ts_IJ = np.array(Ts_IJ)
        # print(time.time() - start)
        # print(Ts_IJ)

        # Ts_IJ = np.apply_along_axis(lambda x: [np.where(x == i)[0] for i in range(3)], 1, episode_batch['info_goal_index'].reshape(rollout_batch_size,T))
        # Ts_IJ = np.array([np.where(episode_batch['info_goal_index'] == i)[1] for i in range(3)])
        # print(Ts_IJ)

        # Select which episodes and time steps to use.
        episode_idxs = np.sort(np.random.randint(0, rollout_batch_size, batch_size))
        t_per_ep = [np.sum(episode_idxs == i) for i in range(rollout_batch_size)]
        # t_samples = np.random.randint(T, size=batch_size)
        # transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       # for key in episode_batch.keys()}
        # print(episode_idxs)
        # print(t_samples)

        # num_candidate_transitions = sum(ep_Ts)
        # print(num_candidate_transitions)
        # num_candidate_transitions = sum(Ts_IJ)
        # print(num_candidate_transitions)

        # #proba of picking transition from each episode for sampling, based on ratio of candidate transitions within episode to total num of candidate transitions
        # probas = [ep_T / num_candidate_transitions for ep_T in ep_Ts]
        # #episode of each sampled transition
        # episode_idxs = np.sort(np.random.choice(rollout_batch_size,batch_size,p=probas))
        # #List denoting how many transitions will be sampled from each corresponding episode according to index
        # t_per_ep = [np.sum(episode_idxs == i) for i in range(rollout_batch_size)]

        #  TAKE t_per_ep SAMPLE FROM EVERY EP AND LINE UP WITH episode_idxs THEN SHUFFLE TOGETHER FOR USE IN CREATING TRANSITIONS
        t_samples = np.array([],dtype=np.int)
        future_offsets = np.array([],dtype=np.int)

        #caclulate HER indices here?

        #TODO: Also see if explicitly enforcing "final" changes much
        for i in range(rollout_batch_size):
            #timesteps of transitions per subgoal for episode i
            ep_ts = Ts_IJ[i]
            #account for single transition sgs to be removed
            mod_T = np.sum([len(sg_t) for sg_t in ep_ts])
            #CANNOT have subgoals with only one timestep be used in ER bc we want to keep it within subgoals
            #Not allowing single transition subgoals into Ts_IJ - see if this causes problems
            probas = [float(len(sg_Ts)) / mod_T for sg_Ts in ep_ts]
            # # print(probas)
            # #randomly sample what subgoals we will use
            sg_indices = np.random.choice(3, t_per_ep[i], p=probas)
            # # print(sg_indices)
            # #number of transitions alloted for each subgoal based on proportion of timesteps
            sg_Ts = [np.sum(sg_indices == i) for i in range(3)]
            # print(sg_Ts)

            for j in range(3):
                sg_T = sg_Ts[j]
                #No samples (covers case if num_sg_ts == 0
                if sg_T == 0:
                    continue
                sg_ts = ep_ts[j]
                num_sg_ts = len(sg_ts)
                # print(j, num_sg_ts)

                # print("j:",j)
                # print(sg_T)

                #sample INDICES OF timesteps to use for ER, don't sample last timestep because want to keep ER within sg
                t_samps_i = np.random.randint(num_sg_ts-1, size=sg_T)# if num_sg_ts > 1 else np.zeros(sg_T).astype(int)
                future_offset_i = np.random.uniform(size=sg_T) * (num_sg_ts-1 - t_samps_i)
                future_offset_i = future_offset_i.astype(int)

                t_samps = sg_ts[t_samps_i]
                #ADDING 1 HERE AND REMOVING 1 BELOW WHEN CALC FUTURE_T, TO STAY WITHIN SG
                future_offset = sg_ts[future_offset_i+t_samps_i+1] - sg_ts[t_samps_i]

                t_samples = np.concatenate((t_samples,t_samps))
                future_offsets = np.concatenate((future_offsets, future_offset))
                # if len(sg_ts) > 0 and sg_ts[-1] - sg_ts[0] != num_sg_ts-1:
                #     print("sg_ts:",sg_ts)
                #     print("t_samps_i",t_samps_i)
                #     print("future_offset_i",future_offset_i)
                #     print("t_samps",t_samps)
                #     print("future_offset",future_offset)

            # #calculate relevant info for corresponding episode
            # t_samps = np.random.randint(ep_Ts[i],size=t_per_ep[i])
            # future_offset = np.random.uniform(size=t_per_ep[i]) * (ep_Ts[i] - t_samps)
            # future_offset = future_offset.astype(int)

            #TODO: does it matter if this is shuffled or not?
            #shuffle all info together
            # inds = np.arange(t_per_ep[i])
            # np.random.shuffle(inds)
            # t_samps = t_samps[inds]
            # # her_inds = her_inds[inds[her_inds]]
            # fut_t = fut_t[inds[her_inds]]
            # fut_ag = fut_ag[inds[her_inds]]

            #concat to output
            # if i == 0:
            #     t_samples = t_samps.copy()
            #     future_offsets = future_offset.copy()
            # else:
                # t_samples = np.concatenate((t_samples,t_samps))
                # future_offsets = np.concatenate((future_offsets,future_offset))

        # print(episode_idxs)
        # print(t_samples)
        # print(future_offsets)
        # print(t_samples + future_offsets)
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        # print(her_indexes)
        # print(rollout_batch_size)
        # print(T)
        #REMOVED ONE HERE AND ADDED ONE ABOVE IN FUTURE_OFFSET CALC
        future_t = (t_samples + future_offsets)[her_indexes]
        # future_t = (t_samples + 1 + future_offsets)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]

        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        #ag_2 used so we can compare achieved goal AFTER that timestep to the actual goal at that step?

        transitions['g'][her_indexes] = future_ag

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        randinds = list(range(transitions['u'].shape[0]))
        random.shuffle(randinds)
        transitions = {k: np.array([transitions[k][i] for i in randinds]) for k in transitions.keys()}

        return transitions

    return _sample_her_transitions
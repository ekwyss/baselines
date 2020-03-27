import numpy as np
import pickle

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

    def _sample_her_transitions(episode_batch, batch_size_in_transitions, ep_Ts):

        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions
        
        # ##Attempt for subgoal based sampling, not using for now
        # # for i in range(rollout_batch_size):
        # #     goal_indices = np.where(episode_batch['info_is_success'][i] == 1)[0]

        # #     cand_indices = [policy_index for policy_index in policy_indexes if policy_index <= len(goal_indices)]
        # #     if len(cand_indices) == 0:
        # #         ep_Ts.append(0)
        # #         continue
        # #     #specific to this bc only 3 subgoals total
        # #     if len(goal_indices) > 2:
        # #         goal_indices = np.concatenate((goal_indices[:2],goal_indices[-1:]))
        # #     else:
        # #         goal_indices = np.concatenate((goal_indices,[T]))
        # #     ep_Ts.append(goal_indices[cand_indices[-1]])

        num_candidate_transitions = sum(ep_Ts)

        #proba of picking transition from each episode for sampling, based on ratio of candidate transitions within episode to total num of candidate transitions
        probas = [ep_T / num_candidate_transitions for ep_T in ep_Ts]
        #episode of each sampled transition
        episode_idxs = np.sort(np.random.choice(rollout_batch_size,batch_size,p=probas))
        #List denoting how many transitions will be sampled from each corresponding episode according to index
        t_per_ep = [np.sum(episode_idxs == i) for i in range(rollout_batch_size)]

        #  TAKE t_per_ep SAMPLE FROM EVERY EP AND LINE UP WITH episode_idxs THEN SHUFFLE TOGETHER FOR USE IN CREATING TRANSITIONS
        t_samples = []
        future_offsets = []

        #TODO: currently using "future" strategy, test against subgoal-based. Also see if explicitly enforcing "final" changes much
        for i in range(rollout_batch_size):
            #calculate relevant info for corresponding episode
            t_samps = np.random.randint(ep_Ts[i],size=t_per_ep[i])
            future_offset = np.random.uniform(size=t_per_ep[i]) * (ep_Ts[i] - t_samps)
            future_offset = future_offset.astype(int)

            #TODO: does it matter if this is shuffled or not?
            #shuffle all info together
            # inds = np.arange(t_per_ep[i])
            # np.random.shuffle(inds)
            # t_samps = t_samps[inds]
            # # her_inds = her_inds[inds[her_inds]]
            # fut_t = fut_t[inds[her_inds]]
            # fut_ag = fut_ag[inds[her_inds]]

            #concat to output
            if i == 0:
                t_samples = t_samps.copy()
                future_offsets = future_offset.copy()
            else:
                t_samples = np.concatenate((t_samples,t_samps))
                future_offsets = np.concatenate((future_offsets,future_offset))

        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        future_t = (t_samples + 1 + future_offsets)[her_indexes]
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

        return transitions

    return _sample_her_transitions
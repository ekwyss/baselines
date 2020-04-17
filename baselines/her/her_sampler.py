import numpy as np
import pickle
# TODO: have both sampler run on every episode and compare
      # also check place where it decides whether to update or not

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
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


    def _sample_her_transitions(episode_batch, batch_size_in_transitions, ts, cand_eps):#, policy_index):#, ep_Ts):

        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        # T = episode_batch['u'].shape[1]
        rollout_batch_size = episode_batch['u'].shape[0]
        batch_size = batch_size_in_transitions

        # ts = np.where(episode_batch['info_goal_index'] == policy_index)
        # ts = [ts[1][np.nonzero(ts[0] == i)[0]] for i in range(rollout_batch_size)]
        ep_Ts = np.array([len(t) for t in ts])
        # if np.sum(ep_Ts) == 0:
        #     return
        # print(ts)

        #prevent episodes with <=1 timesteps dedicaated towards the subgoal to be samples from
        episode_idxs = np.sort(np.random.choice(cand_eps, batch_size))

        # Select which episodes and time steps to use.
        # episode_idxs = np.sort(np.random.randint(0, rollout_batch_size, batch_size))
        t_per_ep = [np.sum(episode_idxs == i) for i in range(rollout_batch_size)]

        t_samples = np.array([],dtype=np.int)
        future_offsets = np.array([],dtype=np.int)

        #caclulate HER indices here?

        #for each episode in batch:
        for i in range(rollout_batch_size):
            t_samples_ep = np.random.randint(ep_Ts[i]-1, size=t_per_ep[i])
            future_offsets_ep = np.random.uniform(size=t_per_ep[i]) * (ep_Ts[i]-1-t_samples_ep)
            future_offsets_ep=future_offsets_ep.astype(int)
            t_samples = np.concatenate((t_samples,ts[i][t_samples_ep]))
            future_offsets = np.concatenate((future_offsets, ts[i][future_offsets_ep+t_samples_ep+1] - ts[i][t_samples_ep]))

        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

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
        # print(transitions['r'])
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        # print("transitions", transitions)
        return transitions

    return _sample_her_transitions



#4/14
    # def _sample_her_transitions(episode_batch, batch_size_in_transitions):
    #     # if batch_size_in_transitions != 49:
    #     # pickle.dump(episode_batch, open("example_episode_3.pkl", "wb"))
    #     #     10/0        
    #     # print(replay_k)
    #     # print(batch_size_in_transitions)
    #     """episode_batch is {key: array(buffer_size x T x dim_key)}
    #     """
    #     T = episode_batch['u'].shape[1]
    #     rollout_batch_size = episode_batch['u'].shape[0]
    #     batch_size = batch_size_in_transitions

    #     # #limit on transition number we can sample from (has to correspond to relevant subgoal)
    #     # ep_Ts = []
        
    #     # ##Attempt for subgoal based sampling, not using for now
    #     # # for i in range(rollout_batch_size):
    #     # #     goal_indices = np.where(episode_batch['info_is_success'][i] == 1)[0]

    #     # #     cand_indices = [policy_index for policy_index in policy_indexes if policy_index <= len(goal_indices)]
    #     # #     if len(cand_indices) == 0:
    #     # #         ep_Ts.append(0)
    #     # #         continue
    #     # #     #specific to this bc only 3 subgoals total
    #     # #     if len(goal_indices) > 2:
    #     # #         goal_indices = np.concatenate((goal_indices[:2],goal_indices[-1:]))
    #     # #     else:
    #     # #         goal_indices = np.concatenate((goal_indices,[T]))
    #     # #     ep_Ts.append(goal_indices[cand_indices[-1]])

    #     # # 3 policy
    #     # for i in range(rollout_batch_size):
    #     #     goal_indices = np.where(episode_batch['info_is_success'][i] == 1)[0]

    #     #     #if don't reach relevant subgoal, don't sample
    #     #     if policy_index > len(goal_indices):
    #     #         ep_Ts.append(0)
    #     #         continue

    #     #     #specific to this bc only 3 subgoals total, see if can get subgoal amount from anywhere
    #     #     if len(goal_indices) > 2:
    #     #         #subgoal indices coincide to first two subgoals reached + last timestep we are still in last subgoal
    #     #         #*but what if we reach last goal, stray out of zone, then return?
    #     #         goal_indices = np.concatenate((goal_indices[:2],goal_indices[-1:]))
    #     #     else:
    #     #         #otherwise we didn't reach the final goal, add goal_indices and total num of timesteps
    #     #         goal_indices = np.concatenate((goal_indices,[T]))
    #     #     #Cap transition number we can sample from for use with relevant subgoal policy
    #     #     ep_Ts.append(goal_indices[policy_index])

    #     #1 policy
    #     # for i in range(rollout_batch_size):
    #     #     ep_Ts.append(T)

    #     num_candidate_transitions = sum(ep_Ts)
        
    #     ##PUT THIS IN POLICIES.PY TO SEE IF SHOULD TRAIN POLICY OR NOT, COMPUTATIONALLY REALLY INEFFICIENT, FIX
    #     ##done, but need to find more efficient way
    #     # if num_candidate_transitions == 0:
    #     #     transitions = {key : np.array([]) for key in episode_batch.keys()}
    #     #     transitions['r'] = np.array([])
    #     #     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
    #     #        for k in transitions.keys()}
    #     #     return transitions
    #     # else:

    #     #proba of picking transition from each episode for sampling, based on ratio of candidate transitions within episode to total num of candidate transitions
    #     probas = [ep_T / num_candidate_transitions for ep_T in ep_Ts]
    #     #episode of each sampled transition
    #     episode_idxs = np.sort(np.random.choice(rollout_batch_size,batch_size,p=probas))
    #     #List denoting how many transitions will be sampled from each corresponding episode according to index
    #     t_per_ep = [np.sum(episode_idxs == i) for i in range(rollout_batch_size)]

    #     # print("policy_index", policy_index)
    #     # print("ep_Ts:", ep_Ts)
    #     # print("num_cand_trans:", num_candidate_transitions)
    #     # print(rollout_batch_size)
    #     # print(batch_size)
    #     # print("probas:", probas)
    #     # print("ep_idxs:", episode_idxs)
    #     # print("t_per_ep:", t_per_ep)
        
    #     #  TAKE t_per_ep SAMPLE FROM EVERY EP AND LINE UP WITH episode_idxs THEN SHUFFLE TOGETHER FOR USE IN CREATING TRANSITIONS
    #     t_samples = []
    #     future_offsets = []
    #     # her_indexes = []
    #     # future_t = []
    #     # future_ag = []
    #     #TODO: currently using "future" strategy, test against subgoal-based. Also see if explicitly enforcing "final" changes much
    #     for i in range(rollout_batch_size):
    #         #calculate relevant info for corresponding episode
    #         t_samps = np.random.randint(ep_Ts[i],size=t_per_ep[i])
    #         # her_inds = np.where(np.random.uniform(size=t_per_ep[i]) < future_p)[0]
    #         future_offset = np.random.uniform(size=t_per_ep[i]) * (ep_Ts[i] - t_samps)
    #         future_offset = future_offset.astype(int)
    #         # her_ind = her_inds.astype(int)
    #         # fut_t = (t_samps + 1 + future_offset)[her_inds]
    #         # fut_ag = episode_batch['ag'][episode_idxs[her_inds], fut_t]

    #         #TODO: does it matter if this is shuffled or not?
    #         #shuffle all info together
    #         # inds = np.arange(t_per_ep[i])
    #         # np.random.shuffle(inds)
    #         # t_samps = t_samps[inds]
    #         # # her_inds = her_inds[inds[her_inds]]
    #         # fut_t = fut_t[inds[her_inds]]
    #         # fut_ag = fut_ag[inds[her_inds]]

    #         #concat to output
    #         if i == 0:
    #             t_samples = t_samps.copy()
    #             future_offsets = future_offset.copy()
    #             # her_indexes = her_inds.copy()
    #             # future_t = fut_t.copy()
    #             # future_ag = fut_ag.copy()
    #         else:
    #             t_samples = np.concatenate((t_samples,t_samps))
    #             future_offsets = np.concatenate((future_offsets,future_offset))
    #             # her_indexes = np.concatenate((her_indexes, her_inds))
    #             # future_t = np.concatenate((future_t, fut_t))
    #             # future_ag = np.concatenate((future_ag,fut_ag))

    #     her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
    #     future_t = (t_samples + 1 + future_offsets)[her_indexes]
    #     future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]

    #     transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
    #                    for key in episode_batch.keys()}

    #     transitions['g'][her_indexes] = future_ag

    #     # Reconstruct info dictionary for reward  computation.
    #     info = {}
    #     for key, value in transitions.items():
    #         if key.startswith('info_'):
    #             info[key.replace('info_', '')] = value

    #     # Re-compute reward since we may have substituted the goal.
    #     reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
    #     reward_params['info'] = info
    #     transitions['r'] = reward_fun(**reward_params)

    #     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
    #                    for k in transitions.keys()}

    #     assert(transitions['u'].shape[0] == batch_size_in_transitions)

    #     return transitions

    # return _sample_her_transitions

############# 2/27 w/o comments #################
# import numpy as np
# import pickle

# def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, policy_index):
#     """Creates a sample function that can be used for HER experience replay.

#     Args:
#         replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
#             regular DDPG experience replay is used
#         replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
#             as many HER replays as regular replays are used)
#         reward_fun (function): function to re-compute the reward with substituted goals
#     """
#     if replay_strategy == 'future':
#         future_p = 1 - (1. / (1 + replay_k))
#     else:  # 'replay_strategy' == 'none'
#         future_p = 0

#     def _sample_her_transitions(episode_batch, batch_size_in_transitions, ep_Ts):
#         T = episode_batch['u'].shape[1]
#         rollout_batch_size = episode_batch['u'].shape[0]
#         batch_size = batch_size_in_transitions

#         num_candidate_transitions = sum(ep_Ts)

#         probas = [ep_T / num_candidate_transitions for ep_T in ep_Ts]
#         episode_idxs = np.sort(np.random.choice(rollout_batch_size,batch_size,p=probas))
#         t_per_ep = [np.sum(episode_idxs == i) for i in range(rollout_batch_size)]

#         #  TAKE t_per_ep SAMPLE FROM EVERY EP AND LINE UP WITH episode_idxs THEN SHUFFLE TOGETHER FOR USE IN CREATING TRANSITIONS
#         t_samples = []
#         future_offsets = []
#         for i in range(rollout_batch_size):
#             #calculate relevant info for corresponding episode
#             t_samps = np.random.randint(ep_Ts[i],size=t_per_ep[i])
#             future_offset = np.random.uniform(size=t_per_ep[i]) * (ep_Ts[i] - t_samps)
#             future_offset = future_offset.astype(int)

#             #concat to output
#             if i == 0:
#                 t_samples = t_samps.copy()
#                 future_offsets = future_offset.copy()
#             else:
#                 t_samples = np.concatenate((t_samples,t_samps))
#                 future_offsets = np.concatenate((future_offsets,future_offset))

#         her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
#         future_t = (t_samples + 1 + future_offsets)[her_indexes]
#         future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]

#         transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
#                        for key in episode_batch.keys()}

#         transitions['g'][her_indexes] = future_ag

#         # Reconstruct info dictionary for reward  computation.
#         info = {}
#         for key, value in transitions.items():
#             if key.startswith('info_'):
#                 info[key.replace('info_', '')] = value

#         # Re-compute reward since we may have substituted the goal.
#         reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
#         reward_params['info'] = info
#         transitions['r'] = reward_fun(**reward_params)

#         transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
#                        for k in transitions.keys()}

#         assert(transitions['u'].shape[0] == batch_size_in_transitions)

#         return transitions

#     return _sample_her_transitions


# ############### ORIGINAL #####################
# import numpy as np


# def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, policy_index):
#     """Creates a sample function that can be used for HER experience replay.
#     Args:
#         replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
#             regular DDPG experience replay is used
#         replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
#             as many HER replays as regular replays are used)
#         reward_fun (function): function to re-compute the reward with substituted goals
#     """
#     if replay_strategy == 'future':
#         future_p = 1 - (1. / (1 + replay_k))
#     else:  # 'replay_strategy' == 'none'
#         future_p = 0

#     def _sample_her_transitions(episode_batch, batch_size_in_transitions):
#         """episode_batch is {key: array(buffer_size x T x dim_key)}
#         """
#         print(batch_size_in_transitions)
#         T = episode_batch['u'].shape[1]
#         rollout_batch_size = episode_batch['u'].shape[0]
#         batch_size = batch_size_in_transitions

#         # Select which episodes and time steps to use.
#         episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
#         t_samples = np.random.randint(T, size=batch_size)
#         transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
#                        for key in episode_batch.keys()}

#         # Select future time indexes proportional with probability future_p. These
#         # will be used for HER replay by substituting in future goals.
#         her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
#         future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
#         future_offset = future_offset.astype(int)
#         future_t = (t_samples + 1 + future_offset)[her_indexes]

#         # Replace goal with achieved goal but only for the previously-selected
#         # HER transitions (as defined by her_indexes). For the other transitions,
#         # keep the original goal.
#         future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
#         transitions['g'][her_indexes] = future_ag

#         # Reconstruct info dictionary for reward  computation.
#         info = {}
#         for key, value in transitions.items():
#             if key.startswith('info_'):
#                 info[key.replace('info_', '')] = value

#         # Re-compute reward since we may have substituted the goal.
#         reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
#         reward_params['info'] = info
#         transitions['r'] = reward_fun(**reward_params)

#         transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
#                        for k in transitions.keys()}

#         assert(transitions['u'].shape[0] == batch_size_in_transitions)

#         return transitions

#     return _sample_her_transitions


######################Testing###########################
# import numpy as np

# def _sample_her_transitions_orig(episode_batch, batch_size_in_transitions, future_p, replay_k, reward_fun, policy_index):
#     """episode_batch is {key: array(buffer_size x T x dim_key)}
#     """
#     T = episode_batch['u'].shape[1]
#     rollout_batch_size = episode_batch['u'].shape[0]
#     batch_size = batch_size_in_transitions
#     np.random.seed(0)

#     # Select which episodes and time steps to use.
#     episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
#     np.random.seed(0)
#     np.random.seed(0)
#     t_samples = np.random.randint(T, size=batch_size)
#     transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
#                    for key in episode_batch.keys()}

#     # Select future time indexes proportional with probability future_p. These
#     # will be used for HER replay by substituting in future goals.
#     np.random.seed(0)
#     her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
#     np.random.seed(0)
#     future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
#     future_offset = future_offset.astype(int)
#     future_t = (t_samples + 1 + future_offset)[her_indexes]

#     # Replace goal with achieved goal but only for the previously-selected
#     # HER transitions (as defined by her_indexes). For the other transitions,
#     # keep the original goal.
#     future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
#     transitions['g'][her_indexes] = future_ag

#     # Reconstruct info dictionary for reward  computation.
#     info = {}
#     for key, value in transitions.items():
#         if key.startswith('info_'):
#             info[key.replace('info_', '')] = value

#     # Re-compute reward since we may have substituted the goal.
#     reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
#     reward_params['info'] = info
#     transitions['r'] = reward_fun(**reward_params)

#     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
#                    for k in transitions.keys()}

#     assert(transitions['u'].shape[0] == batch_size_in_transitions)

#     return transitions

# def _sample_her_transitions_subgoal(episode_batch, batch_size_in_transitions, future_p, replay_k, reward_fun, policy_index):
#     # if batch_size_in_transitions != 49:
#     # pickle.dump(episode_batch, open("example_episode_3.pkl", "wb"))
#     #     10/0        
#     # print(replay_k)
#     # print(batch_size_in_transitions)
#     """episode_batch is {key: array(buffer_size x T x dim_key)}
#     """
#     T = episode_batch['u'].shape[1]
#     rollout_batch_size = episode_batch['u'].shape[0]
#     batch_size = batch_size_in_transitions

#     #limit on transition number we can sample from (has to correspond to relevant subgoal)
#     ep_Ts = []
    
#     ##Attempt for subgoal based sampling, not using for now
#     # for i in range(rollout_batch_size):
#     #     goal_indices = np.where(episode_batch['info_is_success'][i] == 1)[0]

#     #     cand_indices = [policy_index for policy_index in policy_indexes if policy_index <= len(goal_indices)]
#     #     if len(cand_indices) == 0:
#     #         ep_Ts.append(0)
#     #         continue
#     #     #specific to this bc only 3 subgoals total
#     #     if len(goal_indices) > 2:
#     #         goal_indices = np.concatenate((goal_indices[:2],goal_indices[-1:]))
#     #     else:
#     #         goal_indices = np.concatenate((goal_indices,[T]))
#     #     ep_Ts.append(goal_indices[cand_indices[-1]])

#     #3 policy
#     # for i in range(rollout_batch_size):
#     #     goal_indices = np.where(episode_batch['info_is_success'][i] == 1)[0]

#     #     #if don't reach relevant subgoal, don't sample
#     #     if policy_index > len(goal_indices):
#     #         ep_Ts.append(0)
#     #         continue

#     #     #specific to this bc only 3 subgoals total, see if can get subgoal amount from anywhere
#     #     if len(goal_indices) > 2:
#     #         #subgoal indices coincide to first two subgoals reached + last timestep we are still in last subgoal
#     #         #*but what if we reach last goal, stray out of zone, then return?
#     #         goal_indices = np.concatenate((goal_indices[:2],goal_indices[-1:]))
#     #     else:
#     #         #otherwise we didn't reach the final goal, add goal_indices and total num of timesteps
#     #         goal_indices = np.concatenate((goal_indices,[T]))
#     #     #Cap transition number we can sample from for use with relevant subgoal policy
#     #     ep_Ts.append(goal_indices[policy_index])

#     #1 policy
#     for i in range(rollout_batch_size):
#         ep_Ts.append(49)

#     num_candidate_transitions = sum(ep_Ts)
    
#     ##PUT THIS IN POLICIES.PY TO SEE IF SHOULD TRAIN POLICY OR NOT, COMPUTATIONALLY REALLY INEFFICIENT, FIX
#     ##done, but need to find more efficient way
#     # if num_candidate_transitions == 0:
#     #     transitions = {key : np.array([]) for key in episode_batch.keys()}
#     #     transitions['r'] = np.array([])
#     #     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
#     #        for k in transitions.keys()}
#     #     return transitions
#     # else:

#     #proba of picking transition from each episode for sampling, based on ratio of candidate transitions within episode to total num of candidate transitions
#     probas = [ep_T / num_candidate_transitions for ep_T in ep_Ts]
#     np.random.seed(0)
#     #episode of each sampled transition
#     episode_idxs = np.sort(np.random.choice(rollout_batch_size,batch_size,p=probas))
#     #List denoting how many transitions will be sampled from each corresponding episode according to index
#     t_per_ep = [np.sum(episode_idxs == i) for i in range(rollout_batch_size)]

#     # print("policy_index", policy_index)
#     # print("ep_Ts:", ep_Ts)
#     # print("num_cand_trans:", num_candidate_transitions)
#     # print(rollout_batch_size)
#     # print(batch_size)
#     # print("probas:", probas)
#     # print("ep_idxs:", episode_idxs)
#     # print("t_per_ep:", t_per_ep)
    
#     #  TAKE t_per_ep SAMPLE FROM EVERY EP AND LINE UP WITH episode_idxs THEN SHUFFLE TOGETHER FOR USE IN CREATING TRANSITIONS
#     t_samples = []
#     her_indexes = []
#     future_t = []
#     future_ag = []
#     #TODO: currently using "future" strategy, test against subgoal-based. Also see if explicitly enforcing "final" changes much
#     for i in range(rollout_batch_size):
#         #calculate relevant info for corresponding episode
#         np.random.seed(0)
#         t_samps = np.random.randint(ep_Ts[i],size=t_per_ep[i])
#         np.random.seed(0)
#         her_inds = np.where(np.random.uniform(size=t_per_ep[i]) < future_p)[0]
#         np.random.seed(0)
#         future_offset = np.random.uniform(size=t_per_ep[i]) * (ep_Ts[i] - t_samps)
#         future_offset = future_offset.astype(int)
#         her_ind = her_inds.astype(int)
#         fut_t = (t_samps + 1 + future_offset)[her_inds]
#         fut_ag = episode_batch['ag'][episode_idxs[her_inds], fut_t]

#         #TODO: does it matter if this is shuffled or not?
#         #shuffle all info together
#         # inds = np.arange(t_per_ep[i])
#         # np.random.shuffle(inds)
#         # t_samps = t_samps[inds]
#         # # her_inds = her_inds[inds[her_inds]]
#         # fut_t = fut_t[inds[her_inds]]
#         # fut_ag = fut_ag[inds[her_inds]]

#         #concat to output
#         if i == 0:
#             t_samples = t_samps.copy()
#             her_indexes = her_inds.copy()
#             future_t = fut_t.copy()
#             future_ag = fut_ag.copy()
#         else:
#             t_samples = np.concatenate((t_samples,t_samps))
#             her_indexes = np.concatenate((her_indexes, her_inds))
#             future_t = np.concatenate((future_t, fut_t))
#             future_ag = np.concatenate((future_ag,fut_ag))

#     transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
#                    for key in episode_batch.keys()}

#     transitions['g'][her_indexes] = future_ag

#     # Reconstruct info dictionary for reward  computation.
#     info = {}
#     for key, value in transitions.items():
#         if key.startswith('info_'):
#             info[key.replace('info_', '')] = value

#     # Re-compute reward since we may have substituted the goal.
#     reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
#     reward_params['info'] = info
#     transitions['r'] = reward_fun(**reward_params)

#     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
#                    for k in transitions.keys()}

#     assert(transitions['u'].shape[0] == batch_size_in_transitions)

#     return transitions

# import pickle
# def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, policy_index):
#     """Creates a sample function that can be used for HER experience replay.

#     Args:
#         replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
#             regular DDPG experience replay is used
#         replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
#             as many HER replays as regular replays are used)
#         reward_fun (function): function to re-compute the reward with substituted goals
#     """
#     if replay_strategy == 'future':
#         future_p = 1 - (1. / (1 + replay_k))
#     else:  # 'replay_strategy' == 'none'
#         future_p = 0
#     def _sample_her_transitions(episode_batch, batch_size_in_transitions):
#         transitions1 = _sample_her_transitions_orig(episode_batch, batch_size_in_transitions, future_p, replay_k, reward_fun, policy_index)
#         transitions2 = _sample_her_transitions_subgoal(episode_batch, batch_size_in_transitions, future_p, replay_k, reward_fun, policy_index)
#         for thing1, thing2 in zip(transitions1.values(),transitions2.values()):
#             # print(thing1==thing2)
#             same = True
#             for t1,t2 in zip(thing1,thing2):
#                 # print(t1)
#                 if not (t1==t2).all():
#                     same = False
#                     # print(t1,t2)
#             if not same:
#                 pickle.dump(episode_batch, open("example_diff_episode.pkl", "wb"))
#                 print(10/0)
#         print("equal")
#         return transitions1

#     return _sample_her_transitions

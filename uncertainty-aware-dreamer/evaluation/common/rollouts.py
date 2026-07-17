import torch

from uncertainty_aware_dreamer.ssm_mbrl.util.stack_util import stack_maybe_nested_dicts

from evaluation.common.decoding import get_decoded_physical_states


@torch.no_grad()
def get_posteriors(data_collector,
                   start_model_state,
                   start_phys_state,
                   num_rollouts,
                   rollout_length,
                   pred_actions=None,
                   sample_model=True):
    """
    Perform posterior rollouts from the given actions.
    """
    posterior_states = []
    phys_states = []
    observations = []
    actions = []
    rewards = []
    
    if pred_actions is not None:
        # Rollout each individual sequence according to the predicted action sequence.
        for i in range(num_rollouts):
            # infos["state"] = [rollout_length+1 x #phys_dims]
            obs, r, infos, states = data_collector.posterior_rollout_from_actions(max_length=rollout_length,
                                                                                  pred_actions=pred_actions,
                                                                                  start_phys_state=start_phys_state,
                                                                                  start_model_state=start_model_state,
                                                                                  sample_model=sample_model) 
            posterior_states.append(states)
            phys_states.append(infos["state"])
            observations.append(obs)
            rewards.append(r)
            
        # Same action sequence for all rollouts.
        actions = pred_actions.unsqueeze(0).repeat(num_rollouts, 1, 1).to(posterior_states[0]["sample"].device)
    else:
        # Rollout each individual sequence according to policy.
        for i in range(num_rollouts):
            obs, act, r, infos, states = data_collector.rollout_policy(max_length=rollout_length,
                                                                       start_phys_state=start_phys_state,
                                                                       start_model_state=start_model_state,
                                                                       start_action=None,   # use 0 action for start
                                                                       sample_policy=False, # greedy actions
                                                                       sample_model=True)   # act upon sampled model states
            posterior_states.append(states)
            phys_states.append(infos["state"])
            observations.append(obs)
            actions.append(act)
            rewards.append(r)
            
        # Stack over all sequences.
        actions = torch.stack(actions, dim=0)

    posterior_states = stack_maybe_nested_dicts(posterior_states, dim=0)
    phys_states = torch.stack(phys_states, dim=0)
    observations_stacked = [torch.stack(obs, 0) for obs in zip(*observations)]  
    rewards = torch.stack(rewards, dim=0)

    return posterior_states, phys_states, observations_stacked, actions, rewards


@torch.no_grad()
def get_priors(data_collector,
               start_model_state,
               num_rollouts=1,
               include_first=True,
               sample_model=False,
               pred_actions=None):
    """
    Perform prior rollouts from the given actions or purely from imagination, when no actions are given.
    """
    # We can perform the rollouts in parallel.
    if start_model_state["gru_cell_state"].shape[0] == 1:
        # Only one start state given.
        start_model_state = data_collector._model.repeat_state(start_model_state, num_rollouts)
    if pred_actions is not None:
        if len(pred_actions.shape) == 2:
            # Only one action sequence given.
            actions = pred_actions.unsqueeze(0).repeat(num_rollouts, 1, 1).to(start_model_state["sample"].device)
        else:
            actions = pred_actions
        # prior_states include start_model_state_repeat
        prior_states = data_collector.prior_rollout_from_actions(pred_actions=actions,
                                                                 start_model_states=start_model_state,
                                                                 sample_model=sample_model,
                                                                 include_first=include_first) 
    else:
        # prior_states include start_model_state_repeat
        prior_states, _, actions = data_collector.imagine_rollout(start_model_states=start_model_state,
                                                                  sample_policy=False, # greedy actions
                                                                  sample_model=sample_model,
                                                                  include_first=include_first)
            
    return prior_states, actions


def get_one_step_priors(eval_data_collector, posterior_states, gt_actions, rollout_length, num_rollouts, open_loop):
    """
    Compute one-step priors with posterior background, which is the state before being updated with the
    corresponding observation.
    """
    # Enforce one-step imagination only.
    prev_imagine_horizon = eval_data_collector._imagine_horizon
    eval_data_collector._imagine_horizon = 1
    
    prior_posterior_states = []
    first_step = {k: v[:, 0] for k, v in posterior_states.items()}
    if posterior_states["gru_cell_state"].shape[0] == 1 and num_rollouts != 1:
        first_step = eval_data_collector._model.repeat_state(first_step, num_rollouts) 
    prior_posterior_states.append(first_step)
    
    for t in range(rollout_length - 1):
        start_state = {k: v[:, t] for k, v in posterior_states.items()}
        prior_states, _ = get_priors(data_collector=eval_data_collector,
                                     pred_actions=gt_actions[:, t] if open_loop else None,
                                     start_model_state=start_state,
                                     num_rollouts=num_rollouts,
                                     include_first=False,
                                     sample_model=False)
        prior_states = {k: v.squeeze(1) for k, v in prior_states.items()}
        prior_posterior_states.append(prior_states)
        
    eval_data_collector._imagine_horizon = prev_imagine_horizon 

    return stack_maybe_nested_dicts(prior_posterior_states, dim=1)
    

def get_sequence(data_collector, sequence_length, random=False):
    """
    Collect entire sequence with given sequence length.
    """
    all_obs, actions, rewards, infos, post_states = data_collector.collect(sample_policy=False, # greedy actions
                                                                           sample_model=False)  # act upon deterministic model states

    # [num_init_episodes x episode_length+1 x #phys_dims]
    phys_states = torch.stack([info["state"] for info in infos], dim=0)
    
    # We can only sample at most the episode length.
    assert phys_states.shape[1] >= sequence_length
    
    if not random:
        valid_phys_states = phys_states[:, 1:-sequence_length]
        
        # Determine index of posterior states, that maximizes the physical state in any dimension.
        max_dim_indices = torch.argmax(valid_phys_states, dim=-1, keepdim=True)
        max_dim = torch.take_along_dim(valid_phys_states, max_dim_indices, dim=-1).squeeze(-1)
        max_step_indices = torch.argmax(max_dim, dim=-1, keepdim=True)
        max_step = torch.take_along_dim(max_dim, max_step_indices, dim=-1).squeeze(-1)
        
        episode_idx = torch.argmax(max_step, dim=-1)
        start_idx = max_step_indices[episode_idx].item()
    else:
        episode_idx = torch.randint(phys_states.shape[0], (1,)).squeeze(-1)
        start_idx = torch.randint(1, phys_states.shape[1] - sequence_length - 1, (1,)).squeeze(-1)
    
    # Sequence slice for model states and actions.
    seq_slice = slice(start_idx, start_idx + sequence_length)
    # Sequence slice for observations and physical states, since we have a index misalignment due to 0-th obs/info being appended before being associated with the 0-th model state.
    seq_slice_shift = slice(start_idx - 1, start_idx - 1 + sequence_length)

    # Start model state is a dictionary coinciding with the keys of post_states.
    model_states = {k: post_states[episode_idx][k][seq_slice].unsqueeze(0) for k in post_states[0]}
    
    # Start physical state is the physical state at the determined index, which we want to reset the environment to.
    phys_states = infos[episode_idx]["state"][seq_slice_shift].unsqueeze(0)

    # Extract actions.
    actions = actions[episode_idx][seq_slice].unsqueeze(0)
    
    # Extract observation sequence.
    obs = [all_obs[episode_idx][i][seq_slice_shift].unsqueeze(0) for i in range(len(all_obs[0]))]
    
    # Extract rewards.
    rewards = rewards[episode_idx][seq_slice_shift].unsqueeze(0)

    return obs, model_states, phys_states, actions, rewards


def get_random_posteriors_priors(experiment,
                                 data_collector,
                                 rollout_length=50,
                                 num_searches=1000,
                                 verbose=False):
    """
    Collect random posterior and prior rollouts. Priors start at a random model state of the previously
    computed posterior rollout for efficiency reasons.
    """
    def _append_nested(storage, values, squeeze_first=True):
        for key, val in values.items():
            if squeeze_first:
                if isinstance(val, dict):
                    val = {k: v.squeeze(0) for k, v in val.items()}
                else:
                    val = val.squeeze(0)
            storage[key].append(val)
                
    def _stack_nested(storage):
        for key, val in storage.items():
            if isinstance(val[0], dict):
                storage[key] = stack_maybe_nested_dicts(val, dim=0)
            else:
                storage[key] = torch.stack(val, dim=0)
    
    device = experiment._device
    
    prev_imagine_horizon = data_collector._imagine_horizon
    prev_seq_per_collect = data_collector._sequences_per_collect
    data_collector._imagine_horizon = rollout_length
    data_collector._sequences_per_collect = 1
    
    post_infos = {
        "states": [],
        "one_step_prior_states": [],
        "act": [],
        "start_rew": [],
        "start_phys": [],
        "dec_phys": [],
        "dec_phys_one_step": []
    }
    prior_infos = {
        "closed": {
            "states": [],
            "act": [],
            "start_rew": [],
            "start_phys": [],
            "dec_phys": []
        },
        "open": {
            "states": [],
            "act": [],
            "start_rew": [],
            "start_phys": [],
            "dec_phys": []
        }
    }
    for i in range(num_searches):
        # Sample random posterior sequence.
        _, post_states, post_phys, post_acts, post_rew = get_sequence(data_collector=data_collector,
                                                                      sequence_length=rollout_length,
                                                                      random=True)
        post_start_phys = post_phys[0, 0:1]
        post_start_rew = post_rew[0, 0:1]
        
        # Determine corresponding posterior-informed one step priors.
        one_step_prior_states = get_one_step_priors(eval_data_collector=data_collector,
                                                    posterior_states=post_states,
                                                    gt_actions=post_acts,
                                                    rollout_length=rollout_length,
                                                    num_rollouts=1,
                                                    open_loop=False)
        
        # Rollout random closed-loop prior sequence, starting from some random posterior state.
        rand_idx = torch.randint(0, post_phys.shape[0], (1,))
        start_state_closed = {k: v[0, rand_idx] for k, v in post_states.items()}
        prior_start_phys_closed = post_phys[0, rand_idx]
        prior_start_rew_closed = post_rew[0, rand_idx]
        prior_states_closed, prior_acts_closed = get_priors(data_collector=data_collector,
                                                            start_model_state=start_state_closed,
                                                            sample_model=True) # act upon sampled model states
        
        # Rollout open-loop prior sequence according to posterior action sequence
        start_state_open = {k: v[0, 0:1] for k, v in post_states.items()}
        prior_start_phys_open = post_phys[0, 0:1]
        prior_start_rew_open = post_rew[0, 0:1]
        prior_states_open, prior_acts_open = get_priors(data_collector,
                                                       start_model_state=start_state_open,
                                                       pred_actions=post_acts.to(device),
                                                       sample_model=True) # act upon sampled model states
        
        # Get reconstructed physical states.
        post_dec_phys = get_decoded_physical_states(experiment, post_states)
        one_step_prior_dec_phys = get_decoded_physical_states(experiment, one_step_prior_states)
        prior_phys_closed = get_decoded_physical_states(experiment, prior_states_closed)
        prior_phys_open = get_decoded_physical_states(experiment, prior_states_open)
        
        _append_nested(post_infos, {
            "states": post_states,
            "one_step_prior_states": one_step_prior_states,
            "act": post_acts,
            "start_rew": post_start_rew,
            "start_phys": post_start_phys,
            "dec_phys": post_dec_phys,
            "dec_phys_one_step": one_step_prior_dec_phys,
        }, squeeze_first=True)

        _append_nested(prior_infos["closed"], {
            "states": prior_states_closed,
            "act": prior_acts_closed,
            "start_rew": prior_start_rew_closed,
            "start_phys": prior_start_phys_closed,
            "dec_phys": prior_phys_closed,
        }, squeeze_first=True)

        _append_nested(prior_infos["open"], {
            "states": prior_states_open,
            "act": prior_acts_open,
            "start_rew": prior_start_rew_open,
            "start_phys": prior_start_phys_open,
            "dec_phys": prior_phys_open,
        }, squeeze_first=True)
        
        if verbose:
            print("[COLLECT] Finished", i, "searches.")
        
    _stack_nested(post_infos)
    _stack_nested(prior_infos["closed"])
    _stack_nested(prior_infos["open"])
    
    data_collector._imagine_horizon = prev_imagine_horizon
    data_collector._sequences_per_collect = prev_seq_per_collect
    
    all_dict = {
        "post": post_infos,
        "prior": prior_infos
    }
    
    return all_dict
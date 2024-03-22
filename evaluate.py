import gymnasium as gym
import jax
import numpy as np
import equinox as eqx
import argparse
from os.path import join
import pickle

from policy import Policy, QPolicy, EpsPolicy
from config import Config, QMLPConfig
import util


def evaluate(config: Config, seed: int, policy: Policy, repeats: int):
    """
    Produces the mean (and std) reward over `repeats` runs. Uses multiple environments in parallel.
    Note: this method is slightly biased towards runs which finish in fewer steps! But it's fast.
    """
    envs = gym.vector.make(config.env, config.num_envs, asynchronous=True)
    states, info = envs.reset(seed=seed)
    agg_rewards = np.zeros((config.num_envs,), dtype=np.float32)

    finished_rewards = []
    tot_finished = 0

    while tot_finished < repeats:
        actions = policy.get_action(states, None)  # no key given; policy should be the deterministic Q-policy
        actions = np.array(actions)

        states, rewards, terminated, truncated, infos = envs.step(actions)
        agg_rewards += rewards

        if len(infos) > 0:
            finished = np.logical_or(terminated, truncated)

            # Extract finished rewards, reset aggregator
            finished_rewards.append(agg_rewards[finished])
            agg_rewards[finished] = 0

            tot_finished += finished.sum().item()

    envs.close()

    rewards = np.concatenate(finished_rewards)[:tot_finished]
    rewards, std = rewards.mean().item(), rewards.std().item()
    return rewards, std


def evaluate_sequential(config: Config, seed: int, policy: Policy, repeats: int):
    """
    Same as above but runs each simulation sequentially rather than in parallel. Removes the bias towards short runs
    but is ~10x slower.
    """
    env = gym.make(config.env)
    state, _ = env.reset(seed=seed)

    finished_rewards = []
    agg_reward = 0

    key = jax.random.PRNGKey(0)  # won't be used in general; policy will be deterministic Q-policy

    while len(finished_rewards) < repeats:
        key, k = jax.random.split(key)
        action = policy.get_action(state[None], k)[0]
        action = np.array(action)

        state, reward, terminated, truncated, _ = env.step(action)
        agg_reward += float(reward)

        if terminated or truncated:
            seed += 1
            state, _ = env.reset(seed=seed)

            finished_rewards.append(agg_reward)
            agg_reward = 0

    rewards = np.array(finished_rewards)
    rewards, std = rewards.mean().item(), rewards.std().item()
    return rewards, std


def evaluate_multi_agent(config: Config, seed: int, policy: Policy, repeats: int, agent_names):
    """
    Evaluate multi-agent setting. Note vector environments are not supported by PettingZoo so it is sequential.
    """
    env, obs_map = util.make_marl_env(config.env, config.env_kwargs)
    obs_dict, _ = env.reset(seed=seed)
    obs_dict = util.value_map(obs_dict, obs_map)
    gs0 = env.state()

    finished_qvals = []
    finished_rewards = []
    agg_reward = 0
    agg_qval = []
    key = jax.random.PRNGKey(0)  # won't be used in general; policy will be deterministic Q-policy

    while len(finished_rewards) < repeats:
        key, k = jax.random.split(key)
        all_obs = np.concatenate([obs_dict[i][None] for i in agent_names])
        all_actions = policy.get_action(all_obs, k, gstate=gs0)
        qvalue = policy.network.evaluate(all_obs, all_actions, gstate=gs0)  # TODO: unnecessary double network eval
        agg_qval.append(qvalue)

        action_dict = {name: a.item() for name, a in zip(agent_names, all_actions)}

        obs_dict, rewards, terminated, truncated, _ = env.step(action_dict)
        obs_dict = util.value_map(obs_dict, obs_map)
        gs0 = env.state()

        # Take the total reward across agents as the overall reward
        reward = float(sum(rewards.values()))
        agg_reward += reward

        if not env.agents:
            seed += 1
            obs_dict, _ = env.reset()
            obs_dict = util.value_map(obs_dict, obs_map)
            gs0 = env.state()

            finished_rewards.append(agg_reward)
            agg_reward = 0

            mean_qval = sum(agg_qval) / len(agg_qval)
            finished_qvals.append(mean_qval)

    rewards = np.array(finished_rewards)
    rewards, std = rewards.mean().item(), rewards.std().item()
    return rewards, std, sum(finished_qvals) / len(finished_qvals)


def play_single_agent(config: Config, policy: Policy):
    """Plays infinite human visible games."""
    env = gym.make(config.env, render_mode="human")
    state, _ = env.reset(seed=0)
    agg_reward = 0
    while True:
        action = policy.get_action(state[None], key)[0]
        action = np.array(action)
        state, reward, terminated, truncated, _ = env.step(action)
        agg_reward += reward
        if terminated or truncated:
            state, _ = env.reset()

            print(f"Total reward: {agg_reward}")
            agg_reward = 0


def play_multi_agent(config: Config, policy: Policy, agent_names):
    """Plays infinite human visible games."""
    env, obs_map = util.make_marl_env(config.env, config.env_kwargs | {"render_mode": "human"})
    obs_dict, _ = env.reset(seed=0)
    obs_dict = util.value_map(obs_dict, obs_map)
    gs0 = env.state()

    agg_reward = 0

    while True:
        all_obs = np.concatenate([obs_dict[i][None] for i in agent_names])
        all_actions = policy.get_action(all_obs, None, gstate=gs0)

        action_dict = {name: a.item() for name, a in zip(agent_names, all_actions)}

        obs_dict, rewards, terminated, truncated, _ = env.step(action_dict)
        obs_dict = util.value_map(obs_dict, obs_map)
        gs0 = env.state()

        # Take the total reward across agents as the overall reward
        reward = float(sum(rewards.values()))
        agg_reward += reward

        if not env.agents:
            obs_dict, _ = env.reset()
            obs_dict = util.value_map(obs_dict, obs_map)
            gs0 = env.state()

            print(f"Total reward: {agg_reward}")
            agg_reward = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True, help="Path to root directory containing config.json")
    parser.add_argument("-n", "--num_repeats", type=int, default=20)
    parser.add_argument("--max_timestep", type=int, default=-1)

    args = parser.parse_args()

    # Load config from config.json
    root_dir = join("models", args.root)
    config = Config.load(root_dir)

    if args.max_timestep != -1:
        config.env_kwargs["max_timestep"] = args.max_timestep

    is_marl = config.env in util.marl_envs

    # bit of a workaround; construct environment to get its obs/action space shape
    key = jax.random.PRNGKey(0)  # (just used for network init - ovewritten by load)
    if is_marl:
        env, obs_map = util.make_marl_env(config.env, config.env_kwargs)
        agent_names = sorted(env.possible_agents)
        num_agents = len(agent_names)

        obs_shape = obs_map(env.observation_space(agent_names[0]).sample()).shape
        num_actions = env.action_space(agent_names[0]).n

        if hasattr(env, "state_space"):
            global_state_dim = env.state_space.sample().shape[0]
        else:
            global_state_dim = None

        model = config.get_model(obs_shape[0], num_actions, key, num_agents=num_agents, global_state_dim=global_state_dim)
    else:
        env = gym.make(config.env)
        model = config.get_model(env.observation_space.shape[0], env.action_space.n, key)

    env.close()
    model = util.load_model(root_dir, model)

    params, static = eqx.partition(model, eqx.is_array)
    param_count = sum(x.size for x in jax.tree_leaves(params))
    print(f"Parameter count: {param_count}")

    q_policy = QPolicy(model)

    # Load stats
    stats = pickle.load(open(join(root_dir, "stats.pickle"), "rb"))
    util.plot_reward(stats)
    util.plot_loss(stats)

    print("Repeats:", args.num_repeats)

    if is_marl:
        r, s, _ = evaluate_multi_agent(config, 0, q_policy, args.num_repeats, agent_names)
        print(f"Score {r:.4f} +- {s:.3f}")
    else:
        r, s = evaluate(config, 0, q_policy, args.num_repeats)
        print(f"(parallel) Score {r:.2f} +- {s:.1f}")

        r, s = evaluate_sequential(config, 0, q_policy, args.num_repeats)
        print(f"(sequential) Score {r:.2f} +- {s:.1f}")

    # Then run infinite games for fun
    if is_marl:
        play_multi_agent(config, q_policy, agent_names)
    else:
        play_single_agent(config, q_policy)


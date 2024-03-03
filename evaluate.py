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

    while len(finished_rewards) < repeats:
        action = policy.get_action(state[None], None)[0]  # no key given; policy should be the deterministic Q-policy
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

    finished_rewards = []
    agg_reward = 0

    while len(finished_rewards) < repeats:
        all_obs = np.concatenate([obs_dict[i][None] for i in agent_names])
        all_actions = policy.get_action(all_obs, None)

        action_dict = {name: a.item() for name, a in zip(agent_names, all_actions)}

        obs_dict, rewards, terminated, truncated, _ = env.step(action_dict)
        obs_dict = util.value_map(obs_dict, obs_map)

        # Take the mean reward across agents
        reward = sum(rewards.values()) / len(rewards)
        agg_reward += float(reward)

        terminal = bool(env.agents)

        if terminal:
            seed += 1
            obs_dict, _ = env.reset(seed=seed)
            obs_dict = util.value_map(obs_dict, obs_map)

            finished_rewards.append(agg_reward)
            agg_reward = 0

    rewards = np.array(finished_rewards)
    rewards, std = rewards.mean().item(), rewards.std().item()
    return rewards, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True, help="Path to root directory containing config.json")

    args = parser.parse_args()

    # Load config from config.json
    root_dir = join("models", args.root)
    config = Config.load(root_dir)

    env = gym.make(config.env)  # bit of a workaround; construct environment to get its obs/action space shape
    key = jax.random.PRNGKey(0)
    model = config.get_model(env.observation_space.shape[0], env.action_space.n, key)
    env.close()
    model = util.load_model(root_dir, model)

    q_policy = QPolicy(model)

    # Load stats
    stats = pickle.load(open(join(root_dir, "stats.pickle"), "rb"))
    util.plot_reward(stats)
    util.plot_loss(stats)

    # TODO: cmd line arg
    reps = 20
    print("Repeats:", reps)

    r, s = evaluate(config, 0, q_policy, reps)
    print(f"(parallel) Score {r:.2f} +- {s:.1f}")

    r, s = evaluate_sequential(config, 0, q_policy, reps)
    print(f"(sequential) Score {r:.2f} +- {s:.1f}")

    ## Then just runs an infinite sim
    env = gym.make(config.env, render_mode="human")
    state, _ = env.reset(seed=0)
    while True:
        action = q_policy.get_action(state[None], key)[0]
        action = np.array(action)
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()
    env.close()

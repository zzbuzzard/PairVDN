import gymnasium as gym
import jax
import numpy as np
import equinox as eqx
import copy
import argparse
import matplotlib.pyplot as plt
from os.path import join
import pickle
import seaborn as sns
import os

from policy import QPolicy, RandomPolicy
from config import Config
import util
from evaluate import evaluate_multi_agent, evaluate_sequential, evaluate

sns.set_style("whitegrid")

# Multi-agent plots
# roots = ["box_8_rot", "box_16_rot", "box_16_fixed", "simple_spread", "cooking2"]
# repss = [20] * 5
# names = ["IQL", "VDN", "QMIX", "PVDN"]

# Single-agent plot
roots = ["lunar_lander"]
repss = [1] * 3
names = ["4096", "256_256", "128_128_128"]


def get_stuff(stats):
    epochs = sorted(stats["avg_reward"])
    rewards = [stats["avg_reward"][i] for i in epochs]
    err = [stats["std_reward"][i] for i in epochs]
    return epochs, rewards, err


def eval(config, policy, reps):
    is_marl = config.env in util.marl_envs
    print("Running for", reps, "reps")
    if is_marl:
        r, s, _ = evaluate_multi_agent(config, 0, policy, reps, agent_names)
        return r, s
    else:
        r, s = evaluate_sequential(config, 0, policy, reps)
        return r, s


for reps, root in zip(repss, roots):
    # if root not in ["box_16_rot", "box_16_fixed"]:
    #     continue
    print()
    print("ENVIRONMENT", root)
    print()
    box = "box" in root

    all_stats = {}
    final_reward = {}
    reward_plot = {}
    if box:
        reward_1000 = {}

    for name in names:
        jax.clear_caches()

        root_dir = join("models", root, name)
        if not os.path.isdir(root_dir):
            continue
        config = Config.load(root_dir)
        is_marl = config.env in util.marl_envs

        key = jax.random.PRNGKey(0)  # (just used for network init - ovewritten by load)
        if is_marl:
            env, obs_map = util.make_marl_env(config.env, config.env_kwargs)
            agent_names = sorted(env.possible_agents)
            num_agents = len(agent_names)

            obs_shape = obs_map(env.observation_space(agent_names[0]).sample()).shape
            num_actions = env.action_space(agent_names[0]).n

            action_space = env.action_space(agent_names[0])

            if hasattr(env, "state_space"):
                global_state_dim = env.state_space.sample().shape[0]
            else:
                global_state_dim = None

            model = config.get_model(obs_shape[0], num_actions, key, num_agents=num_agents, global_state_dim=global_state_dim)
        else:
            env = gym.make(config.env)
            action_space = env.action_space
            model = config.get_model(env.observation_space.shape[0], env.action_space.n, key)

        env.close()
        model = util.load_model(root_dir, model)
        q_policy = QPolicy(model)
        stats = pickle.load(open(join(root_dir, "stats.pickle"), "rb"))

        params, static = eqx.partition(model, eqx.is_array)
        param_count = sum(x.size for x in jax.tree_leaves(params))
        print(f"Parameter count of {name} is {param_count}")

        all_stats[name] = stats
        final_reward[name] = eval(config, q_policy, reps)
        reward_plot[name] = get_stuff(stats)

        if box:
            old = copy.deepcopy(config.env_kwargs)
            config.env_kwargs["max_timestep"] = 1000
            reward_1000[name] = eval(config, q_policy, reps)
            config.env_kwargs = old

    baseline = RandomPolicy(action_space)
    r, s = eval(config, baseline, 20)
    if box:
        config.env_kwargs["max_timestep"] = 1000
        r_1000, s_1000 = eval(config, baseline, reps)

    def epoch_to_buf_size(epoch):
        tot = config.simulation_steps_initial + epoch * config.simulation_steps_per_epoch
        return min(tot, config.exp_buffer_len)

    def epochs_to_steps(epoch):
        t = 0
        for i in range(1, epoch+1):
            t += epoch_to_buf_size(i)
        return t // 32

    print()
    print("SCORES:")
    print(f"RAND:\t\t{r:.4f} +- {s:.3f}")
    for name in names:
        if name in final_reward:
            r1, s1 = final_reward[name]
            print(f"{name}:\t\t{r1:.4f} +- {s1:.3f}")
    if box:
        print("SCORES (1000!):")
        print(f"RAND:\t\t{r_1000:.4f} +- {s_1000:.3f}")
        for name in names:
            r2, s2 = reward_1000[name]
            print(f"{name}:\t\t{r2:.4f} +- {s2:.3f}")

    ######## LOSS PLOT
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for idx, name in enumerate(reward_plot):
        if name not in reward_plot:
            continue
        col = f"C{idx}"
        stats = all_stats[name]
        epochs = sorted(stats["loss"])
        losses = [stats["loss"][i] for i in epochs]
        ax.plot(epochs, losses, label=name, color=col)
    ax.legend(loc='upper left')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(f"pics/{root}_loss.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()
    quit(0)

    ######### REWARD PLOT
    # fig, ax = plt.subplots(figsize=(5, 3.5))
    fig, ax = plt.subplots(figsize=(6, 4.2))

    for idx, name in enumerate(reward_plot):
        if name not in reward_plot:
            continue
        epochs, rewards, err = reward_plot[name]
        col = f"C{idx}"

        mod_epochs = [epochs_to_steps(i) for i in epochs]
        ax.plot(mod_epochs, rewards, label=name, color=col)
        ax.fill_between(mod_epochs, [r - e for r, e in zip(rewards, err)], [r + e for r, e in zip(rewards, err)],
                         color=col,
                         linewidth=0, alpha=0.3)

    ax.axhline(y=r, color='black', linestyle='--')
    ax.legend(loc='upper left')
    ax.set_xlabel("Batches Seen")
    ax.set_ylabel("Average Reward")
    plt.tight_layout()

    plt.savefig(f"pics/{root}.pdf", bbox_inches='tight', pad_inches=0.05)
    plt.show()

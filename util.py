import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from os.path import join, isfile
from typing import Tuple, Dict, Callable
import pettingzoo
from gymnasium import spaces
from pettingzoo.butterfly import cooperative_pong_v5, knights_archers_zombies_v10 # , pistonball_v6 # TODO
from pettingzoo.sisl import pursuit_v4
from pettingzoo.mpe import simple_spread_v3
import numpy as np
from cooking_zoo import environment as cookenv

from box_env import BoxJumpEnvironment


def save_model(root_path: str, model: eqx.Module):
    path = join(root_path, "model.eqx")
    print("Serialising model to", path)
    eqx.tree_serialise_leaves(path, model)


def load_model(root_path: str, model: eqx.Module):
    path = join(root_path, "model.eqx")
    if isfile(path):
        print("Deserialising from", path)
        return eqx.tree_deserialise_leaves(path, model)
    else:
        print(path, "does not exist, not loading model")
        return model


def tree_zip_map(f, tree1, tree2):
    """
    For two trees with the *same shape*, constructs a new tree where each leaf is a function of the two corresponding
    leaves.
    """
    flat1, treedef = jax.tree_flatten(tree1)
    flat2, _ = jax.tree_flatten(tree2)
    flat_out = [f(i, j) for i, j in zip(flat1, flat2)]
    return jax.tree_unflatten(treedef, flat_out)


def plot_reward(stats):
    epochs = sorted(stats["avg_reward"])
    rewards = [stats["avg_reward"][i] for i in epochs]
    err = [stats["std_reward"][i] for i in epochs]

    plt.plot(epochs, rewards)
    plt.fill_between(epochs, [r - e for r, e in zip(rewards, err)], [r + e for r, e in zip(rewards, err)], color='C0',
                     linewidth=0, alpha=0.3)
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss(stats):
    epochs = sorted(stats["loss"])
    losses = [stats["loss"][i] for i in epochs]
    plt.plot(epochs, losses)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()


def key_map(x: Dict, f) -> Dict:
    return {f(u): v for u, v in x.items()}


def value_map(x: Dict, f) -> Dict:
    return {u: f(v) for u, v in x.items()}


marl_envs = ["cooperative_pong", "knights_archers_zombies", "pursuit", "simple_spread"]


def make_marl_env(name: str, env_kwargs: dict) -> Tuple[pettingzoo.ParallelEnv, Callable]:
    """
    PettingZoo doesn't have a pettingzoo.make() apparently, but also I have custom settings for many of these.
    Returns an environment and a function which simplifies the observation (e.g. flattens it)
    """
    if name == "cooperative_pong":
        remove_channel = lambda obs: np.mean(obs, axis=2)
        return cooperative_pong_v5.parallel_env(cake_paddle=False, **env_kwargs), remove_channel
    elif name == "knights_archers_zombies":
        return (knights_archers_zombies_v10.parallel_env(killable_knights=False, killable_archers=False, **env_kwargs),
                np.ndarray.flatten)
    elif name == "pursuit":
        # Note: surround=False is a much easier version of this game
        # Note: (7 x 7 x 3) spatial input, CNN arch would be better than flattening
        return pursuit_v4.parallel_env(**env_kwargs), np.ndarray.flatten
    elif name == "simple_spread":
        return simple_spread_v3.parallel_env(**env_kwargs), lambda x: x
    elif name == "cooking":
        return make_cooking_env(mode=1, **env_kwargs)
    elif name == "cooking2":
        return make_cooking_env(mode=2, **env_kwargs)
    elif name == "boxjump":
        return BoxJumpEnvironment(**env_kwargs), lambda x: x
    else:
        raise NotImplementedError(f"Unknown MARL environment '{name}'.")


def make_cooking_env(mode=1, num_agents=2, max_steps=400, render_mode="", **env_kwargs):
    render = render_mode == "human"
    obs_spaces = ["feature_vector"] * num_agents

    recipes = ["TomatoLettuceSalad", "CarrotBanana"]

    agent_visualization = ["human"] * num_agents
    reward_scheme = {"recipe_reward": 20, "max_time_penalty": -5, "recipe_penalty": -20, "recipe_node_reward": 5}  # 0->5
    action_scheme = "scheme3"
    if mode == 1:
        level = "coop_test"
        meta_file = "example"
    elif mode == 2:
        level = "simple"
        meta_file = "meta_small"

    env = cookenv.cooking_env.parallel_env(level=level, meta_file=meta_file, num_agents=num_agents, max_steps=max_steps,
                                           recipes=recipes, agent_visualization=agent_visualization,
                                           obs_spaces=obs_spaces, end_condition_all_dishes=True,
                                           action_scheme=action_scheme, render=render, reward_scheme=reward_scheme)

    old_state_fn = env.state
    env.state = lambda: old_state_fn().astype(np.float32)
    return env, lambda x: x.astype(np.float32)


def update_parameter(layer, name, new_value):
    return eqx.tree_at(lambda l: getattr(l, name), layer, new_value)


def update_parameters(layer, names, new_values):
    for name, new_value in zip(names, new_values):
        layer = eqx.tree_at(lambda l: getattr(l, name), layer, new_value)
    return layer


def small_init(linear_layer, mul=0.01, zero_bias=True):
    new_weight = linear_layer.weight * mul
    if zero_bias:
        new_bias = jnp.zeros_like(linear_layer.bias)
        return update_parameters(linear_layer, ["weight", "bias"], [new_weight, new_bias])
    else:
        return update_parameter(linear_layer, "weight", new_weight)

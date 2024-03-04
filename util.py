import jax
import equinox as eqx
import matplotlib.pyplot as plt
from os.path import join, isfile
from typing import Tuple, Dict, Callable
import pettingzoo
from pettingzoo.butterfly import cooperative_pong_v5, knights_archers_zombies_v10 # , pistonball_v6 # TODO
from pettingzoo.sisl import pursuit_v4
from pettingzoo.mpe import simple_spread_v3
import numpy as np

from network import QFunc


def save_model(root_path: str, model: eqx.Module):
    path = join(root_path, "model.eqx")
    print("Serialising model to", path)
    eqx.tree_serialise_leaves(path, model)


def load_model(root_path: str, model: eqx.Module) -> QFunc:
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
    else:
        raise NotImplementedError(f"Unknown MARL environment '{name}'.")

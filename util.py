import jax
import equinox as eqx
import matplotlib.pyplot as plt
from os.path import join, isfile

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

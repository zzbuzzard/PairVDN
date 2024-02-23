"""Implements the DQN"""
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import equinox as eqx
from equinox import nn
from optax import adam
import matplotlib.pyplot as plt
from tqdm import tqdm
from jaxtyping import Array, Float, PyTree
from typing import List


# Just an MLP with ReLU activation; simple DQN implementation.
class QMLP(eqx.Module):
    layers: List

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int], key):
        assert len(hidden_layers) >= 1
        super().__init__()
        activation = jax.nn.relu

        sizes = [input_dim] + hidden_layers + [output_dim]
        layers = []

        for i in range(len(sizes) - 1):
            key, key2 = random.split(key)
            layers.append(nn.Linear(sizes[i], sizes[i+1], key=key2))
            layers.append(activation)

        self.layers = layers[:-1]  # remove final activation

    def __call__(self, x):
        for f in self.layers:
            x = f(x)
        return x

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

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int], final_layer_small_init: bool, key):
        assert len(hidden_layers) >= 1
        super().__init__()
        activation = jax.nn.relu

        sizes = [input_dim] + hidden_layers + [output_dim]
        layers = []

        for i in range(len(sizes) - 1):
            key, key2 = random.split(key)
            layers.append(nn.Linear(sizes[i], sizes[i+1], key=key2))
            layers.append(activation)

        self.layers = layers[:-1]  # remove trailing activation

        if final_layer_small_init:
            # Initialise final layer with smaller values (adapted from equinox docs under 'tricks')
            lin = self.layers[-1]
            new_weight = lin.weight * 0.01
            new_bias = jnp.zeros_like(lin.bias)
            lin2 = eqx.tree_at(lambda l: l.bias,
                               eqx.tree_at(lambda l: l.weight, lin, new_weight),
                               new_bias)
            self.layers[-1] = lin2

    def __call__(self, x):
        for f in self.layers:
            x = f(x)
        return x

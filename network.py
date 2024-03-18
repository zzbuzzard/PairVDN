"""Implements DQN"""
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float, PyTree
from typing import List
from abc import ABC, abstractmethod

import util


class QFunc(eqx.Module, ABC):
    """
    Abstract class for parameterisations of Q-functions. A Q-function must provide argmax and evaluate
    interfaces. This class is shared between the simple single-agent case and the multi-agent decomposition
    networks.

    single-agent case: obs, actions are from a batch of single-agent environments
    multi-agent case: obs, actions are from a single multi-agent environment with one entry per agent
    """

    @abstractmethod
    def argmax(self, obs, gstate=None):
        """Return argmax actions given this observation. Formally, argmax_a Q(obs, a)"""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, obs, actions, gstate=None):
        """Return Q(obs, actions)"""
        raise NotImplementedError

    def max(self, obs, gstate=None):
        """Return max_a Q(obs, a). Defaults to evaluate(argmax())"""
        actions = self.argmax(obs)
        return self.evaluate(obs, actions)


# Just an MLP with ReLU activation; simple DQN implementation.
class QMLP(QFunc):
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
            self.layers[-1] = util.small_init(self.layers[-1], mul=0.01)

    @eqx.filter_jit
    def __call__(self, x):
        for f in self.layers:
            x = f(x)
        return x

    def argmax(self, obs, gstate=None):
        q_values = vmap(self.__call__)(obs)  # B x A
        return jnp.argmax(q_values, axis=-1)  # B

    def evaluate(self, obs, actions, gstate=None):
        q_values = vmap(self.__call__)(obs)  # B x A
        batch_size = obs.shape[0]  # = B
        return q_values[jnp.arange(batch_size), actions]  # B

    def max(self, obs, gstate=None):
        q_values = vmap(self.__call__)(obs)  # B x A
        return jnp.max(q_values, axis=-1)  # B

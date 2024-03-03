"""Implements VDN etc"""
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float, PyTree
from typing import List
from abc import ABC, abstractmethod

from network import QMLP


class MultiQ(eqx.Module, ABC):
    @abstractmethod
    def argmax(self, obs):
        """Return argmax actions given this observation. Formally, argmax_a Q(obs, a)"""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, obs, actions):
        """Return Q(obs, actions)"""
        raise NotImplementedError

    def max(self, obs):
        """Return max_a Q(obs, a). Defaults to evaluate(argmax())"""
        actions = self.argmax(obs)
        return self.evaluate(obs, actions)


# Just an MLP with ReLU activation; simple DQN implementation.
class VDN(MultiQ):
    qs: List
    num_agents: int

    def __init__(self, num_agents: int, key, **kwargs):
        self.num_agents = num_agents
        keys = jax.random.split(key, num_agents)
        self.qs = [QMLP(**kwargs, key=k) for k in keys]

    # TODO: Idk why i can't call this
    def compile(self):
        for q in self.qs:
            q.__call__ = eqx.filter_jit(q.__call__)

    def argmax(self, obs):
        actions = []
        for i in range(self.num_agents):
            actions.append(jnp.argmax(self.qs[i](obs[i])))
        return jnp.array(actions)

    def max(self, obs):
        t = 0
        for i in range(self.num_agents):
            t += jnp.max(self.qs[i](obs[i]))
        return t

    def evaluate(self, obs, actions):
        t = 0
        for i in range(self.num_agents):
            t += self.qs[i](obs[i])[actions[i]]
        return t


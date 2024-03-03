"""Implements VDN etc"""
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float, PyTree
from typing import List

from network import QMLP, QFunc


class VDN(QFunc):
    qs: List
    num_agents: int

    def __init__(self, num_agents: int, key, **kwargs):
        self.num_agents = num_agents
        keys = jax.random.split(key, num_agents)
        self.qs = [QMLP(**kwargs, key=k) for k in keys]

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


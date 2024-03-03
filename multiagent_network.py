"""Implements VDN etc"""
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float, PyTree
from typing import List

from network import QMLP, QFunc


class IndividualQ(QFunc):
    """
    IQL baseline. This is kind of a special case, as the other classes here are decomposition networks i.e.
    represent a single joint Q function. Max/evaluate, for this class only, return lists not single Q-values.

    Conveniently, this works in my MARL framework without changes - the batched loss function returns a list
    of (B x N) values rather than (B) values, so taking the overall loss as the mean still works.
    """
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
        qs = []
        for i in range(self.num_agents):
            qs.append(jnp.max(self.qs[i](obs[i])))
        return jnp.array(qs)

    def evaluate(self, obs, actions):
        qs = []
        for i in range(self.num_agents):
            qs.append(self.qs[i](obs[i])[actions[i]])
        return jnp.array(qs)


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


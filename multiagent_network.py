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
    share_params: bool

    def __init__(self, num_agents: int, share_params: bool, key, **kwargs):
        self.num_agents = num_agents
        self.share_params = share_params

        if share_params:
            self.qs = [QMLP(**kwargs, key=key)]
        else:
            keys = jax.random.split(key, num_agents)
            self.qs = [QMLP(**kwargs, key=k) for k in keys]

    # get ith implicit Q-network
    def gq(self, idx):
        if self.share_params:
            return self.qs[0]
        return self.qs[idx]

    def argmax(self, obs):
        actions = []
        for i in range(self.num_agents):
            actions.append(jnp.argmax(self.gq(i)(obs[i])))
        return jnp.array(actions)

    def max(self, obs):
        qs = []
        for i in range(self.num_agents):
            qs.append(jnp.max(self.gq(i)(obs[i])))
        return jnp.array(qs)

    def evaluate(self, obs, actions):
        qs = []
        for i in range(self.num_agents):
            qs.append(self.gq(i)(obs[i])[actions[i]])
        return jnp.array(qs)


class VDN(QFunc):
    qs: List
    num_agents: int
    share_params: bool

    def __init__(self, num_agents: int, share_params: bool, key, **kwargs):
        self.num_agents = num_agents
        self.share_params = share_params

        if share_params:
            self.qs = [QMLP(**kwargs, key=key)]
        else:
            keys = jax.random.split(key, num_agents)
            self.qs = [QMLP(**kwargs, key=k) for k in keys]

    # get ith implicit Q-network
    def gq(self, idx):
        if self.share_params:
            return self.qs[0]
        return self.qs[idx]

    def argmax(self, obs):
        actions = []
        for i in range(self.num_agents):
            actions.append(jnp.argmax(self.gq(i)(obs[i])))
        return jnp.array(actions)

    def max(self, obs):
        t = 0
        for i in range(self.num_agents):
            t += jnp.max(self.gq(i)(obs[i]))
        return t

    def evaluate(self, obs, actions):
        t = 0
        for i in range(self.num_agents):
            t += self.gq(i)(obs[i])[actions[i]]
        return t


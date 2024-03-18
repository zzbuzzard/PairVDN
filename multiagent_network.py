"""Implements VDN etc"""
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float, PyTree
from typing import List

import util
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

    def argmax(self, obs, gstate=None):
        actions = []
        for i in range(self.num_agents):
            actions.append(jnp.argmax(self.gq(i)(obs[i])))
        return jnp.array(actions)

    def max(self, obs, gstate=None):
        qs = []
        for i in range(self.num_agents):
            qs.append(jnp.max(self.gq(i)(obs[i])))
        return jnp.array(qs)

    def evaluate(self, obs, actions, gstate=None):
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

    def argmax(self, obs, gstate=None):
        actions = []
        for i in range(self.num_agents):
            actions.append(jnp.argmax(self.gq(i)(obs[i])))
        return jnp.array(actions)

    def max(self, obs, gstate=None):
        t = 0
        for i in range(self.num_agents):
            t += jnp.max(self.gq(i)(obs[i]))
        return t

    def evaluate(self, obs, actions, gstate=None):
        t = 0
        for i in range(self.num_agents):
            t += self.gq(i)(obs[i])[actions[i]]
        return t


class QMIX(QFunc):
    qs: List
    num_agents: int
    hidden_dim: int
    share_params: bool

    # Hypernets
    mk_w1: eqx.Module
    mk_w2: eqx.Module
    mk_b1: eqx.Module
    mk_b2: List

    def __init__(self, num_agents: int, share_params: bool, state_dim: int, hidden_dim: int, key, **kwargs):
        # Note gstate refers to the *global* not per-agent state (these are called 'obs')
        self.num_agents = num_agents
        self.share_params = share_params
        self.hidden_dim = hidden_dim

        if share_params:
            key, k = jax.random.split(key)
            self.qs = [QMLP(**kwargs, key=k)]
        else:
            keys = jax.random.split(key, num_agents + 1)
            self.qs = [QMLP(**kwargs, key=k) for k in keys]
            key = keys[-1]

        # Define HyperNets:
        # W2(W1 Qs + B1) + B2
        # W1 : num_agents x hidden_dim
        # B1 : hidden_dim
        # W2 : hidden_dim x 1
        # B2 : 1
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.mk_w1 = nn.Linear(state_dim, out_features=(num_agents * hidden_dim), key=k1)
        self.mk_b1 = nn.Linear(state_dim, out_features=hidden_dim, key=k2)
        self.mk_w2 = nn.Linear(state_dim, out_features=(hidden_dim * 1), key=k3)
        self.mk_b2 = [nn.Linear(state_dim, hidden_dim//2, key=k4), nn.Linear(hidden_dim//2, 1, key=k5)]

    # get ith implicit Q-network
    def gq(self, idx):
        if self.share_params:
            return self.qs[0]
        return self.qs[idx]

    def argmax(self, obs, gstate=None):
        assert gstate is not None

        actions = []
        for i in range(self.num_agents):
            actions.append(jnp.argmax(self.gq(i)(obs[i])))
        return jnp.array(actions)

    @eqx.filter_jit
    def _eval(self, qs, gstate):
        """Apply the non-monotonic aggregation to N Q-values in the list `qs`"""
        w1 = jnp.abs(self.mk_w1(gstate)).reshape((self.num_agents, self.hidden_dim))
        w2 = jnp.abs(self.mk_w2(gstate))
        b1 = self.mk_b1(gstate)
        b2 = self.mk_b2[1](jax.nn.relu(self.mk_b2[0](gstate)))

        xs = (qs @ w1) + b1  # num_agents -> hidden_dim
        xs = jax.nn.relu(xs)
        q_out = (xs @ w2) + b2  # hidden_dim -> scalar

        return q_out

    def max(self, obs, gstate=None):
        assert gstate is not None

        qs = []
        for i in range(self.num_agents):
            qs.append(jnp.max(self.gq(i)(obs[i])))
        qs = jnp.array(qs)

        return self._eval(qs, gstate)

    def evaluate(self, obs, actions, gstate=None):
        assert gstate is not None

        qs = []
        for i in range(self.num_agents):
            qs.append(self.gq(i)(obs[i])[actions[i]])
        qs = jnp.array(qs)

        return self._eval(qs, gstate)

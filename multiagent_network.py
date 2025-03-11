"""Implements VDN etc"""
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from equinox import nn
from jaxtyping import Array, Float, PyTree
from typing import List
import numpy as np
import math

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

    @eqx.filter_jit
    def argmax(self, obs, gstate=None):
        actions = []
        for i in range(self.num_agents):
            actions.append(jnp.argmax(self.gq(i)(obs[i])))
        return jnp.array(actions)

    @eqx.filter_jit
    def max(self, obs, gstate=None):
        t = 0
        for i in range(self.num_agents):
            t += jnp.max(self.gq(i)(obs[i]))
        return t

    @eqx.filter_jit
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
        self.mk_b2 = [nn.Linear(state_dim, hidden_dim, key=k4), nn.Linear(hidden_dim, 1, key=k5)]

        # HyperNetwork initialisation as in https://arxiv.org/pdf/2312.08399.pdf
        w1_weight = (3 / (num_agents * state_dim)) ** 0.5
        w2_weight = (3 / (hidden_dim * state_dim)) ** 0.5
        b1_weight = (3 / (2 * state_dim)) ** 0.5
        b2_weight = (3 / (2 * hidden_dim)) ** 0.5
        self.mk_w1 = util.custom_init(self.mk_w1, w1_weight, 0, key=k1)
        self.mk_b1 = util.custom_init(self.mk_b1, b1_weight, 0, key=k2)
        self.mk_w2 = util.custom_init(self.mk_w2, w2_weight, 0, key=k3)
        self.mk_b2[1] = util.custom_init(self.mk_b2[1], b2_weight, 0, key=k5)

        # The weights of these layers dictate how important the state is for influencing these parameters
        # The biases are the initial weights, so should be initialised in the usual way
        # weight_size = (1 / state_dim**0.5)  # (start with low importance)
        # bias_size_w1 = bias_size_b1 = 1 / (num_agents ** 0.5)
        # bias_size_w2 = bias_size_b2 = 1 / (hidden_dim ** 0.5)
        # self.mk_w1 = util.custom_init(self.mk_w1, weight_size, bias_size_w1, key=k1)
        # self.mk_b1 = util.custom_init(self.mk_b1, weight_size, bias_size_b1, key=k2)
        # self.mk_w2 = util.custom_init(self.mk_w2, weight_size, bias_size_w2, key=k3)
        # self.mk_b2[0] = util.small_init(self.mk_b2[0], zero_bias=True)
        # self.mk_b2[1] = util.custom_init(self.mk_b2[1], weight_size, bias_size_b2, key=k5)

        # self.mk_w1 = util.small_init(self.mk_w1, zero_bias=False)
        # self.mk_b1 = util.small_init(self.mk_b1, zero_bias=False)
        # self.mk_w2 = util.small_init(self.mk_w2, zero_bias=False)
        # self.mk_b2[0] = util.small_init(self.mk_b2[0], zero_bias=False)

    # get ith implicit Q-network
    def gq(self, idx):
        if self.share_params:
            return self.qs[0]
        return self.qs[idx]

    @eqx.filter_jit
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

    @eqx.filter_jit
    def max(self, obs, gstate=None):
        assert gstate is not None

        qs = []
        for i in range(self.num_agents):
            qs.append(jnp.max(self.gq(i)(obs[i])))
        qs = jnp.array(qs)

        return self._eval(qs, gstate)

    @eqx.filter_jit
    def evaluate(self, obs, actions, gstate=None):
        assert gstate is not None

        qs = []
        for i in range(self.num_agents):
            qs.append(self.gq(i)(obs[i])[actions[i]])
        qs = jnp.array(qs)

        return self._eval(qs, gstate)


class PairVDN(QFunc):
    qs: List
    share_params: bool

    # static allows use in jitted functions
    num_agents: int = eqx.field(static=True)
    action_space_size: int = eqx.field(static=True)
    order: np.ndarray = eqx.field(static=True)
    inv_order: np.ndarray = eqx.field(static=True)

    def __init__(self, num_agents: int, share_params: bool, random_order: bool, key, **kwargs):
        self.num_agents = num_agents
        self.share_params = share_params

        # Order is a permutation array converting from the given order (alphabetical) to the local order
        if random_order:
            self.order = np.random.permutation(num_agents)
        else:
            self.order = np.arange(num_agents)
        self.inv_order = util.inverse_permutation(self.order)

        self.action_space_size = kwargs["output_dim"]

        # Pair Q-functions are Q(s1, s2) = [q values for all action pairs] so double input, squared output
        kwargs["input_dim"] = kwargs["input_dim"] * 2
        kwargs["output_dim"] = kwargs["output_dim"] ** 2

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

    # Time: O(NA^3)
    @eqx.filter_jit
    def _maximise(self, q_out):
        """
        :param q_out: List of outputs of (Q1, Q2, ...). Each has size |A|^2
        :return: Tuple: (optimal action sequence, Q-score of optimal action sequence)
        """
        asize = self.action_space_size
        num_agent = self.num_agents

        # D : asize x num_agent x asize
        #  D(a1, n, a2) = max score achievable with agent 1 choosing a1 and agent n choosing an
        #  Dprev stores backtracking information to retrieve optimal a sequence
        D = jnp.zeros((asize, num_agent, asize))
        Dprev = jnp.zeros((asize, num_agent, asize), dtype=np.int32)

        # D(a1, 1, a2) = Q1(a1, a2)
        for a1 in range(asize):
            for a2 in range(asize):
                D = D.at[a1, 1, a2].set(q_out[0, a1, a2])
                Dprev = Dprev.at[a1, 1, a2].set(a1)

        # D(a1, n, a(n+1)) = max [ D(a1, n-1, an) + Qn(an, a(n+1)) ]
        for a1 in range(asize):
            for i in range(2, num_agent):
                for a in range(asize):
                    scores = D[a1, i - 1] + q_out[i - 1, :, a]  # broadcasting over (an)
                    D = D.at[a1, i, a].set(jnp.max(scores))
                    Dprev = Dprev.at[a1, i, a].set(jnp.argmax(scores))

        # Complete the loop: calculate best score achievable for each a1 via
        #  score[a1] = max aN. D(a1, N-1, aN) + QN(aN, a1)
        scores = jnp.zeros((asize,))
        ans = jnp.zeros((asize,), dtype=np.int32)
        for a1 in range(asize):
            # all_scores[an] = D(a1, -1, an) + Qn(an, a1)
            all_scores = D[a1, -1, :] + q_out[-1, :, a1]
            scores = scores.at[a1].set(jnp.max(all_scores))
            ans = ans.at[a1].set(jnp.argmax(all_scores))

        a1 = jnp.argmax(scores)
        an = ans[a1]

        # Finally, backtrack to produce the optimal sequence a_s
        a_s = jnp.zeros((num_agent,), dtype=np.int32)
        a_s = a_s.at[-1].set(an)
        for i in range(num_agent - 2, -1, -1):
            a_s = a_s.at[i].set(Dprev[a1, i + 1, a_s[i + 1]])

        return a_s, scores[a1]

    # Time: O(N A^N)
    # Naive maximisation algorithm to sanity check the efficient DP one
    def _maximise_naive(self, q_out):
        """
        :param q_out: List of outputs of (Q1, Q2, ...). Each has size |A|^2
        :return: Tuple: (optimal action sequence, Q-score of optimal action sequence)
        """

        def score(a_s):
            t = 0
            for i in range(num_agent):
                j = (i + 1) % num_agent
                t += q_out[i, a_s[i], a_s[j]]
            return t

        best_t = -math.inf
        best_as = None

        asize = self.action_space_size
        num_agent = self.num_agents

        for i in range(asize ** num_agent):
            a_s = []
            for j in range(num_agent):
                a_s.append(i % asize)
                i //= asize

            t = score(a_s)
            if t > best_t:
                best_t = t
                best_as = a_s

        return best_as, best_t

    @eqx.filter_jit()
    def _get_q_out(self, obs):
        obs = obs[self.order]  # permute to local order

        q_out = []
        for i in range(self.num_agents):
            j = (i + 1) % self.num_agents
            o1, o2 = obs[i], obs[j]
            o = jnp.concatenate((o1, o2))
            # Reshape raw output of MLP to an |A| x |A| grid
            out = self.gq(i)(o).reshape((self.action_space_size, self.action_space_size))
            q_out.append(out[None])
        return jnp.concatenate(q_out)

    @eqx.filter_jit
    def argmax(self, obs, gstate=None):
        q_out = self._get_q_out(obs)
        argmax, qmax = self._maximise(q_out)
        return argmax[self.inv_order]  # permute back to global order

    @eqx.filter_jit
    def max(self, obs, gstate=None):
        q_out = self._get_q_out(obs)
        argmax, qmax = self._maximise(q_out)
        return qmax

    @eqx.filter_jit
    def evaluate(self, obs, actions, gstate=None):
        obs = obs[self.order]  # permute to local order
        actions = actions[self.order]

        t = 0
        for i in range(self.num_agents):
            j = (i + 1) % self.num_agents
            o1, o2 = obs[i], obs[j]
            o = jnp.concatenate((o1, o2))
            # Reshape raw output of MLP to an |A| x |A| grid
            out = self.gq(i)(o).reshape((self.action_space_size, self.action_space_size))
            t += out[actions[i], actions[j]]
        return t


class PairWVDN(QFunc):
    qs: List
    share_params: bool

    # static allows use in jitted functions
    num_agents: int = eqx.field(static=True)
    action_space_size: int = eqx.field(static=True)
    order: np.ndarray = eqx.field(static=True)
    inv_order: np.ndarray = eqx.field(static=True)

    hidden_dim: int
    weight_mlp: eqx.Module

    def __init__(self, num_agents: int, share_params: bool, state_dim: int, hidden_dim: int, random_order: bool, key,
                 **kwargs):
        self.num_agents = num_agents
        self.share_params = share_params
        self.hidden_dim = hidden_dim

        # Order is a permutation array converting from the given order (alphabetical) to the local order
        if random_order:
            self.order = np.random.permutation(num_agents)
        else:
            self.order = np.arange(num_agents)
        self.inv_order = util.inverse_permutation(self.order)

        self.action_space_size = kwargs["output_dim"]

        # Pair Q-functions are Q(s1, s2) = [q values for all action pairs] so double input, squared output
        kwargs["input_dim"] = kwargs["input_dim"] * 2
        kwargs["output_dim"] = kwargs["output_dim"] ** 2

        k1, k2, key = jax.random.split(key, num=3)
        self.weight_mlp = nn.Sequential([nn.Linear(state_dim, out_features=hidden_dim, key=k1),
                                         nn.Linear(hidden_dim, out_features=num_agents, key=k2)])

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

    # Time: O(NA^3)
    @eqx.filter_jit
    def _maximise(self, q_out):
        """
        :param q_out: List of outputs of (Q1, Q2, ...). Each has size |A|^2. Pre-multiplied by weights.
        :return: Tuple: (optimal action sequence, Q-score of optimal action sequence)
        """
        asize = self.action_space_size
        num_agent = self.num_agents

        # D : asize x num_agent x asize
        #  D(a1, n, a2) = max score achievable with agent 1 choosing a1 and agent n choosing an
        #  Dprev stores backtracking information to retrieve optimal a sequence
        D = jnp.zeros((asize, num_agent, asize))
        Dprev = jnp.zeros((asize, num_agent, asize), dtype=np.int32)

        # D(a1, 1, a2) = Q1(a1, a2)
        for a1 in range(asize):
            for a2 in range(asize):
                D = D.at[a1, 1, a2].set(q_out[0, a1, a2])
                Dprev = Dprev.at[a1, 1, a2].set(a1)

        # D(a1, n, a(n+1)) = max [ D(a1, n-1, an) + Qn(an, a(n+1)) ]
        for a1 in range(asize):
            for i in range(2, num_agent):
                for a in range(asize):
                    scores = D[a1, i - 1] + q_out[i - 1, :, a]  # broadcasting over (an)
                    D = D.at[a1, i, a].set(jnp.max(scores))
                    Dprev = Dprev.at[a1, i, a].set(jnp.argmax(scores))

        # Complete the loop: calculate best score achievable for each a1 via
        #  score[a1] = max aN. D(a1, N-1, aN) + QN(aN, a1)
        scores = jnp.zeros((asize,))
        ans = jnp.zeros((asize,), dtype=np.int32)
        for a1 in range(asize):
            # all_scores[an] = D(a1, -1, an) + Qn(an, a1)
            all_scores = D[a1, -1, :] + q_out[-1, :, a1]
            scores = scores.at[a1].set(jnp.max(all_scores))
            ans = ans.at[a1].set(jnp.argmax(all_scores))

        a1 = jnp.argmax(scores)
        an = ans[a1]

        # Finally, backtrack to produce the optimal sequence a_s
        a_s = jnp.zeros((num_agent,), dtype=np.int32)
        a_s = a_s.at[-1].set(an)
        for i in range(num_agent - 2, -1, -1):
            a_s = a_s.at[i].set(Dprev[a1, i + 1, a_s[i + 1]])

        return a_s, scores[a1]

    # Time: O(N A^N)
    # Naive maximisation algorithm to sanity check the efficient DP one
    def _maximise_naive(self, q_out):
        """
        :param q_out: List of outputs of (Q1, Q2, ...). Each has size |A|^2
        :return: Tuple: (optimal action sequence, Q-score of optimal action sequence)
        """

        def score(a_s):
            t = 0
            for i in range(num_agent):
                j = (i + 1) % num_agent
                t += q_out[i, a_s[i], a_s[j]]
            return t

        best_t = -math.inf
        best_as = None

        asize = self.action_space_size
        num_agent = self.num_agents

        for i in range(asize ** num_agent):
            a_s = []
            for j in range(num_agent):
                a_s.append(i % asize)
                i //= asize

            t = score(a_s)
            if t > best_t:
                best_t = t
                best_as = a_s

        return best_as, best_t

    @eqx.filter_jit()
    def _get_q_out(self, obs, gstate):
        obs = obs[self.order]  # permute to local order

        # jax.nn.sigmoid
        weights = self.weight_mlp[1](jax.nn.leaky_relu(self.weight_mlp[0](gstate)))

        q_out = []
        for i in range(self.num_agents):
            j = (i + 1) % self.num_agents
            o1, o2 = obs[i], obs[j]
            o = jnp.concatenate((o1, o2))
            # Reshape raw output of MLP to an |A| x |A| grid
            out = self.gq(i)(o).reshape((self.action_space_size, self.action_space_size))
            q_out.append(out[None])

        return jnp.concatenate(q_out) * weights[:, None, None]

    @eqx.filter_jit
    def argmax(self, obs, gstate):
        q_out = self._get_q_out(obs, gstate)
        argmax, qmax = self._maximise(q_out)
        return argmax[self.inv_order]  # permute back to global order

    @eqx.filter_jit
    def max(self, obs, gstate):
        q_out = self._get_q_out(obs, gstate)
        argmax, qmax = self._maximise(q_out)
        return qmax

    @eqx.filter_jit
    def evaluate(self, obs, actions, gstate):
        obs = obs[self.order]  # permute to local order
        actions = actions[self.order]

        weights = self.weight_mlp[1](jax.nn.leaky_relu(self.weight_mlp[0](gstate)))

        t = 0
        for i in range(self.num_agents):
            j = (i + 1) % self.num_agents
            o1, o2 = obs[i], obs[j]
            o = jnp.concatenate((o1, o2))
            # Reshape raw output of MLP to an |A| x |A| grid
            out = self.gq(i)(o).reshape((self.action_space_size, self.action_space_size))
            t += out[actions[i], actions[j]] * weights[i]
        return t

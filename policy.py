import gymnasium as gym
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from abc import ABC, abstractmethod
from network import QFunc


class Policy(ABC):
    """
    Abstract policy which acts on batches of states. Action_space should be the *single* action space.
    In the single-agent setting, `states` is a batch of single-agent environments.
    In the multi-agent setting, `states` is a single multi-agent environment containing per-agent observations.
    """
    @abstractmethod
    def get_action(self, states, key):
        """Returns batch of actions"""
        raise NotImplementedError


class QPolicy(Policy):
    def __init__(self, network: QFunc):
        self.network = network

    @eqx.filter_jit
    def get_action(self, states, key):
        return self.network.argmax(states)


# Decorator pattern aw yeah
class EpsPolicy(Policy):
    def __init__(self, action_space: gym.Space, policy: Policy, eps: float):
        self.action_space = action_space
        self.policy = policy
        self.eps = eps

    @eqx.filter_jit
    def get_action(self, states, key):
        n = states.shape[0]

        key, k1, k2, k3 = jax.random.split(key, 4)
        use_random = jax.random.uniform(k1, (n,), minval=0, maxval=1) < self.eps

        # Random actions
        random_action = jax.random.randint(k2, (n,), 0, self.action_space.n, dtype=self.action_space.dtype)
        # Q-policy actions
        q_action = self.policy.get_action(states, k3)

        out = jnp.where(use_random, random_action, q_action)

        return out

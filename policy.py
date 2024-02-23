import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod


class Policy(ABC):
    """Abstract policy which acts on batches of states. Action_space should be the *single* action space."""
    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    @abstractmethod
    def get_action(self, states, key):
        """Returns actions and new key"""
        raise NotImplementedError


class QPolicy(Policy):
    def __init__(self, action_space: gym.Space, network):
        super().__init__(action_space)
        self.network = network

    def get_action(self, states, key):
        q_vals = jax.vmap(self.network)(states)
        return jnp.argmax(q_vals, axis=0)


class EpsPolicy(Policy):
    def __init__(self, action_space: gym.Space, policy: Policy, eps: float):
        super().__init__(action_space)
        self.policy = policy
        self.eps = eps

    def get_action(self, states, key):
        n = states.shape[0]

        key, k1, k2, k3 = jax.random.split(key, 4)
        use_random = jax.random.uniform(k1, (n,), minval=0, maxval=1) < self.eps

        # Random actions
        out = jax.random.randint(k2, (n,), 0, self.action_space.n, dtype=self.action_space.dtype)
        # Insert policy actions at non-random locations
        out = out.at[~use_random].set(self.policy.get_action(states[~use_random], k3))

        return out

import jax
import equinox as eqx

import util


class TargetNetwork:
    def __init__(self, network: eqx.Module, gamma: float = 0.99):
        self.network = network
        self.gamma = gamma

    def update(self, network: eqx.Module):
        def combine(old_param, new_param):
            return old_param * self.gamma + new_param * (1 - self.gamma)

        old_params, old_static = eqx.partition(self.network, eqx.is_array)
        params, static = eqx.partition(network, eqx.is_array)

        new_params = util.tree_zip_map(combine, old_params, params)

        self.network = eqx.combine(new_params, old_static)

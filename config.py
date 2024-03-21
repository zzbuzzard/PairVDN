from dataclasses import dataclass, asdict, field
from os import path
import json
from typing import List
from abc import ABC

import network
import multiagent_network


@dataclass
class ModelConfig(ABC):
    pass


@dataclass
class QMLPConfig(ModelConfig):
    hidden_layers: List[int]


@dataclass
class VDNConfig(ModelConfig):
    hidden_layers: List[int]
    share_params: bool = False


@dataclass
class QMIXConfig(ModelConfig):
    hidden_layers: List[int]
    hidden_dim: int
    share_params: bool = False


@dataclass
class IQLConfig(ModelConfig):
    hidden_layers: List[int]
    share_params: bool = False


@dataclass
class PairVDNConfig(ModelConfig):
    hidden_layers: List[int]
    share_params: bool = False
    random_order: bool = False


@dataclass
class Config:
    model_config: ModelConfig
    model_type: str

    env: str = "LunarLander-v2"
    seed: int = 0

    # Generic training stuff
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    display_every: int = 20
    save_every: int = 20
    eval_every: int = 5
    eval_reps: int = 8
    opt: str = "Adam"

    # RL training params
    exp_buffer_len: int = 100000
    gamma: float = 0.99
    target_network_gamma: float = 0.99
    exploration_eps_start: float = 1.0
    exploration_eps_end: float = 0.05
    exploration_lerp_epochs: int = -1  # Number of epochs to interpolate exploration over; -1 for all
    simulation_steps_initial: int = 0  # number of random experiences to add at the start
    simulation_steps_per_epoch: int = 1000  # creates simulation_steps_per_epoch * num_envs datapoints
    num_envs: int = 16
    final_layer_small_init: bool = False

    # MARL stuff
    env_kwargs: dict = field(default_factory=dict)

    @staticmethod
    def load(root_path: str):
        """
        Loads a Config object from [root_path]/config.json.
        """
        jsonpath = path.join(root_path, "config.json")
        assert path.isdir(root_path), f"Model directory '{root_path}' not found!"
        assert path.isfile(jsonpath), f"config.json not found in directory '{root_path}'."

        with open(jsonpath, "r") as file:
            data = json.loads(file.read())

        if data["model_type"] == "QMLP":
            data["model_config"] = QMLPConfig(**data["model_config"])
        elif data["model_type"] == "VDN":
            data["model_config"] = VDNConfig(**data["model_config"])
        elif data["model_type"] == "QMIX":
            data["model_config"] = QMIXConfig(**data["model_config"])
        elif data["model_type"] == "IQL":
            data["model_config"] = IQLConfig(**data["model_config"])
        elif data["model_type"] == "PairVDN":
            data["model_config"] = PairVDNConfig(**data["model_config"])
        else:
            raise NotImplementedError(f"Unknown model type '{data['model_type']}'")

        return Config(**data)

    def get_model(self, input_dim, output_dim, key, global_state_dim=None, num_agents=None):
        if self.model_type == "QMLP":
            return network.QMLP(input_dim, output_dim, self.model_config.hidden_layers, self.final_layer_small_init,
                                key)
        elif self.model_type == "VDN":
            return multiagent_network.VDN(num_agents, self.model_config.share_params, key, input_dim=input_dim,
                                          output_dim=output_dim, hidden_layers=self.model_config.hidden_layers,
                                          final_layer_small_init=self.final_layer_small_init)
        elif self.model_type == "QMIX":
            assert global_state_dim is not None
            return multiagent_network.QMIX(num_agents, self.model_config.share_params, global_state_dim,
                                           self.model_config.hidden_dim,
                                           key, input_dim=input_dim, output_dim=output_dim,
                                           hidden_layers=self.model_config.hidden_layers,
                                           final_layer_small_init=self.final_layer_small_init)
        elif self.model_type == "IQL":
            return multiagent_network.IndividualQ(num_agents, self.model_config.share_params, key, input_dim=input_dim,
                                                  output_dim=output_dim,
                                                  hidden_layers=self.model_config.hidden_layers,
                                                  final_layer_small_init=self.final_layer_small_init)
        elif self.model_type == "PairVDN":
            return multiagent_network.PairVDN(num_agents, self.model_config.share_params,
                                              self.model_config.random_order,
                                              key, input_dim=input_dim, output_dim=output_dim,
                                              hidden_layers=self.model_config.hidden_layers,
                                              final_layer_small_init=self.final_layer_small_init)
        else:
            raise NotImplementedError(f"Unknown model '{self.model_type}'.")

    def get_eps(self, epoch):
        end = self.num_epochs if self.exploration_lerp_epochs == -1 else self.exploration_lerp_epochs
        if epoch >= end:
            epoch = end
        a = (epoch - 1) / (end - 1)  # (epoch is 1-based)
        return self.exploration_eps_start + a * (self.exploration_eps_end - self.exploration_eps_start)

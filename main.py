import gymnasium as gym
from whatis import whatis as wi
import jax
import jax.numpy as jnp
import numpy as np
from replay_buffer import ExperienceBuffer, batched_dataloader

# TODO: Cmd line args, or config
exp_buffer_len = 100000
batch_size = 32
num_envs = 16
name = "LunarLander-v2"
seed = 0
simulation_steps_per_epoch = 1000  # creates simulation_steps_per_epoch * num_envs datapoints
num_epochs = 100

num_runs = 0


def collect_data(envs: gym.Env, policy, buffer: ExperienceBuffer, steps: int):
    global num_runs
    states, info = envs.reset(seed=seed + num_runs)
    num_runs += 1

    for _ in range(steps):
        # TODO: use Policy
        actions = envs.action_space.sample()

        next_states, rewards, terminated, truncated, infos = envs.step(actions)

        # The vector environment automatically resets environments when they finish. To prevent next_state = the new start
        #  state, we have to use the "final_observation" field in the info dict.
        if len(infos) > 0:
            finished = np.logical_or(terminated, truncated)
            # Note: we don't want to *modify* next_states, as we need the start states for the reset things to be kept
            next_states_without_restart = next_states.copy()
            next_states_without_restart[finished] = np.vstack(infos["final_observation"][finished])
        else:
            next_states_without_restart = next_states

        buffer.add_experiences(state=states, next_state=next_states_without_restart, action=actions, reward=rewards)

        states = next_states


# env = gym.make(name, render_mode="human")
envs = gym.vector.make(name, num_envs, asynchronous=False)  # TODO: Turn on Async once everything else is working

state_shape = envs.single_observation_space.sample().shape
action_shape = envs.action_space.sample().shape
buffer = ExperienceBuffer(exp_buffer_len,
                          keys=["state", "next_state", "action", "reward"],
                          key_shapes=[state_shape, state_shape, action_shape, tuple()],
                          key_dtypes=[np.float32, np.float32, np.int64, np.float32])

for epoch in range(num_epochs):
    # Collect some nice fresh data
    collect_data(envs, None, buffer, simulation_steps_per_epoch)
    

envs.close()

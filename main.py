import gymnasium as gym
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from optax import adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from os.path import join

from replay_buffer import ExperienceBuffer, batched_dataloader
from policy import Policy, QPolicy, EpsPolicy
from config import Config
import util

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--root", type=str, required=True, help="Path to root directory containing config.json")

args = parser.parse_args()

# Load config from config.json
root_dir = join("models", args.root)
config = Config.load(root_dir)

num_runs = 0


def collect_data(key, envs: gym.Env, policy: Policy, buffer: ExperienceBuffer, steps: int):
    global num_runs
    states, info = envs.reset(seed=config.seed + num_runs)
    num_runs += 1

    for _ in range(steps):
        key, k1 = random.split(key)
        actions = policy.get_action(states, k1)
        actions = np.array(actions)

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

    return key


def loss_single(model: eqx.Module, s0, s1, a, r):
    """Computes loss on *non-batched* data"""
    a0_scores = model(s0)
    q1 = a0_scores[a]

    a1_scores = model(s1)
    q_max = jnp.max(a1_scores)
    q2 = r + config.gamma * q_max

    return (q1 - q2) ** 2


loss_batched = jax.vmap(loss_single, in_axes=(None, 0, 0, 0, 0))


@eqx.filter_value_and_grad
def loss_fn(model, s0, s1, a, r):
    """Computes loss on batched data"""
    losses = loss_batched(model, s0, s1, a, r)
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(model: eqx.Module, opt_state, s0, s1, a, r):
    # Note: operates on batched data!
    loss_val, grad = loss_fn(model, s0, s1, a, r)
    updates, opt_state = opt.update(grad, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


envs = gym.vector.make(config.env, config.num_envs, asynchronous=True)

state_shape = envs.single_observation_space.sample().shape
action_shape = envs.single_action_space.sample().shape
buffer = ExperienceBuffer(config.exp_buffer_len,
                          keys=["state", "next_state", "action", "reward"],
                          key_shapes=[state_shape, state_shape, action_shape, tuple()],
                          key_dtypes=[np.float32, np.float32, np.int64, np.float32])

# Load model
key = random.PRNGKey(config.seed)
key, k1 = random.split(key)
model = config.get_model(state_shape[0], envs.single_action_space.n, k1)
print("Model:")
print(model)
print()

# Load optimiser
opt = adam(config.learning_rate)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

it = tqdm(range(config.num_epochs))
for epoch in it:
    q_policy = QPolicy(model)
    q_policy.get_action = eqx.filter_jit(q_policy.get_action)

    eps = config.get_eps(epoch)
    eps_policy = EpsPolicy(envs.single_action_space, q_policy, eps)

    # Collect some nice fresh data
    key = collect_data(key, envs, eps_policy, buffer, config.simulation_steps_per_epoch)

    dl = batched_dataloader(buffer, batch_size=config.batch_size, drop_last=True)

    epoch_losses = []
    for batch in dl:
        s0 = jnp.asarray(batch["state"])
        s1 = jnp.asarray(batch["next_state"])
        a = jnp.asarray(batch["action"])
        r = jnp.asarray(batch["reward"])

        model, opt_state, loss_val = train_step(model, opt_state, s0, s1, a, r)
        epoch_losses.append(loss_val.item())

    avg_loss = sum(epoch_losses) / len(epoch_losses)
    it.set_description(f"Loss = {avg_loss:.2f}")

    if epoch > 0 and epoch % config.save_every == 0:
        util.save_model(root_dir, model)

    if epoch > 0 and epoch % config.display_every == 0:
        q_policy = QPolicy(model)
        q_policy.get_action = eqx.filter_jit(q_policy.get_action)

        env = gym.make(config.env, render_mode="human")
        state, _ = env.reset(seed=epoch)
        for _ in range(1000):
            action = q_policy.get_action(state[None], key)[0]
            action = np.array(action)
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                state, _ = env.reset()
        env.close()

# Save final model
util.save_model(root_dir, model)

envs.close()

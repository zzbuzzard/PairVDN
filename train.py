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
import pickle
import wandb
from dataclasses import asdict

from replay_buffer import ExperienceBuffer, batched_dataloader
from policy import Policy, QPolicy, EpsPolicy
from config import Config
import util
import evaluate
from target_network import TargetNetwork


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

        terminal = np.logical_or(terminated, truncated)  # 'd' flag for each environment

        # Note: next_state is *invalid* when terminal=true, as it actually gives the starting state for the next run.
        #  however, the DQN loss function does not use the next state when terminal=true so this is ok.
        buffer.add_experiences(state=states, next_state=next_states, action=actions, reward=rewards, terminal=terminal)

        states = next_states

    return key


def loss_single(model: eqx.Module, target_model: eqx.Module, s0, s1, a, r, d):
    """Computes loss on *non-batched* data"""
    a0_scores = model(s0)
    q1 = a0_scores[a]

    a1_scores = target_model(s1)
    q_max = jnp.max(a1_scores)
    q2 = r + (1 - d) * config.gamma * q_max  # 1 - d -> nullifies when d=1

    return (q1 - q2) ** 2


loss_batched = jax.vmap(loss_single, in_axes=(None, None, 0, 0, 0, 0, 0))


@eqx.filter_value_and_grad
def loss_fn(model, target_model, s0, s1, a, r, d):
    """Computes loss on batched data"""
    losses = loss_batched(model, target_model, s0, s1, a, r, d)
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(model: eqx.Module, opt_state, target_model: eqx.Module, s0, s1, a, r, d):
    loss_val, grad = loss_fn(model, target_model, s0, s1, a, r, d)
    updates, opt_state = opt.update(grad, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, required=True, help="Path to root directory containing config.json")
    args = parser.parse_args()

    # Load config from config.json
    root_dir = join("models", args.root)
    config = Config.load(root_dir)

    wandb.init(
        project="RL_Project",
        name=args.root,
        config=asdict(config)
    )

    envs = gym.vector.make(config.env, config.num_envs, asynchronous=True)

    state_shape = envs.single_observation_space.sample().shape
    action_shape = envs.single_action_space.sample().shape
    buffer = ExperienceBuffer(config.exp_buffer_len,
                              keys=["state", "next_state", "action", "reward", "terminal"],
                              key_shapes=[state_shape, state_shape, action_shape, tuple(), tuple()],
                              key_dtypes=[np.float32, np.float32, np.int64, np.float32, np.bool_])

    # Load model
    np.random.seed(config.seed)
    key = random.PRNGKey(config.seed)
    key, k1 = random.split(key)
    model = config.get_model(state_shape[0], envs.single_action_space.n, k1)
    print("Model:")
    print(model)
    print()

    target_model = TargetNetwork(model, config.target_network_gamma)

    # Load optimiser
    opt = adam(config.learning_rate)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # Track training stats
    stats = {"avg_reward": {}, "std_reward": {}, "loss": {}}

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
            d = jnp.asarray(batch["terminal"]).astype(np.float32)

            model, opt_state, loss_val = train_step(model, opt_state, target_model.network, s0, s1, a, r, d)
            epoch_losses.append(loss_val.item())

            target_model.update(model)

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        it.set_description(f"Loss = {avg_loss:.2f}")

        stats["loss"][epoch] = avg_loss
        wandb.log({"loss": avg_loss, "epoch": epoch})

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

        if epoch > 0 and epoch % config.eval_every == 0:
            q_policy = QPolicy(model)
            q_policy.get_action = eqx.filter_jit(q_policy.get_action)

            avg_reward, std_reward = evaluate.evaluate(config, seed=epoch, policy=q_policy, repeats=config.eval_reps)
            stats["avg_reward"][epoch] = avg_reward
            stats["std_reward"][epoch] = std_reward

            wandb.log({"reward": avg_reward, "epoch": epoch})

    # Save final model
    util.save_model(root_dir, model)

    envs.close()

    # Save stats
    pickle.dump(stats, open(join(root_dir, "stats.pickle"), "wb"))

    util.plot_reward(stats)
    util.plot_loss(stats)

    wandb.finish()

import pettingzoo
from pettingzoo.butterfly import cooperative_pong_v5
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
from whatis import whatis as wi
import time

from train import loss_batched
from replay_buffer import ExperienceBuffer, batched_dataloader
from policy import Policy, QPolicy, EpsPolicy
from config import Config
import util
import evaluate
from target_network import TargetNetwork
from network import QFunc


num_runs = 0


def collect_data(key, env: pettingzoo.ParallelEnv, policy: Policy, buffer: ExperienceBuffer, steps: int):
    global num_runs
    obs_dict, _ = env.reset(seed=config.seed + num_runs)
    obs_dict = util.value_map(obs_dict, obs_map)
    num_runs += 1

    for _ in range(steps):
        key, k1 = random.split(key)
        all_obs = jnp.concatenate([obs_dict[i][None] for i in agent_names])
        all_actions = policy.get_action(all_obs, k1)

        action_dict = {name: a.item() for name, a in zip(agent_names, all_actions)}

        next_obs_dict, rewards, terminated, truncated, _ = env.step(action_dict)
        next_obs_dict = util.value_map(next_obs_dict, obs_map)

        # Take the total reward across agents as the overall reward
        reward = float(sum(rewards.values()))

        terminal = not bool(env.agents)

        all_next_obs = np.concatenate([next_obs_dict[i][None] for i in agent_names])
        reward = np.array([reward])[0]
        terminal = np.array([terminal], dtype=np.bool_)[0]

        buffer.add_experiences(all_obs=all_obs, all_next_obs=all_next_obs, all_actions=all_actions, reward=reward,
                               terminal=terminal, single_mode=True)

        obs_dict = next_obs_dict

        if terminal:
            obs_dict, _ = env.reset(seed=config.seed + num_runs)
            obs_dict = util.value_map(obs_dict, obs_map)
            num_runs += 1

    return key


# The version in train() operates on batches of single-agent environments, i.e. shape (B x ObsShape)
# Here during training we operate on batches of multi-agent environments, i.e. shape (B x N x ObsShape)
#  so we require another vmap
loss_batched_batched = jax.vmap(loss_batched, in_axes=(None, None, 0, 0, 0, 0, 0, None))


@eqx.filter_value_and_grad
def loss_fn(model: QFunc, target_model: QFunc, s0, s1, a, r, d, gamma):
    """Computes loss on batched data"""
    losses = loss_batched_batched(model, target_model, s0, s1, a, r, d, gamma)
    return jnp.mean(losses)


@eqx.filter_jit
def train_step(model: QFunc, opt_state, target_model: QFunc, s0, s1, a, r, d, gamma):
    loss_val, grad = loss_fn(model, target_model, s0, s1, a, r, d, gamma)
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
        config=asdict(config),
        tags=["MARL"]
    )

    env, obs_map = util.make_marl_env(config.env, config.env_kwargs)

    agent_names = sorted(env.possible_agents)  # sort for determinacy
    agent_name_to_index = {i: name for i, name in enumerate(agent_names)}

    n = len(agent_names)
    obs_shape = obs_map(env.observation_space(agent_names[0]).sample()).shape
    action_shape = env.action_space(agent_names[0]).sample().shape
    num_actions = env.action_space(agent_names[0]).n

    all_obs_shape = (n,) + obs_shape
    all_actions_shape = (n,) + action_shape

    buffer = ExperienceBuffer(config.exp_buffer_len,
                              keys=["all_obs", "all_next_obs", "all_actions", "reward", "terminal"],
                              key_shapes=[all_obs_shape, all_obs_shape, all_actions_shape, tuple(), tuple()],
                              key_dtypes=[np.float32, np.float32, np.int64, np.float32, np.bool_])

    # Load model
    np.random.seed(config.seed)
    key = random.PRNGKey(config.seed)
    key, k1 = random.split(key)
    model = config.get_model(obs_shape[0], num_actions, k1, num_agents=n)
    print("Model:")
    print(model)
    print()

    target_model = TargetNetwork(model, config.target_network_gamma)

    # Load optimiser
    opt = adam(config.learning_rate)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # Track training stats
    stats = {"avg_reward": {}, "std_reward": {}, "loss": {}}

    it = tqdm(range(1, config.num_epochs + 1))
    for epoch in it:
        q_policy = QPolicy(model)

        eps = config.get_eps(epoch)
        eps_policy = EpsPolicy(env.action_space(agent_names[0]), q_policy, eps)

        # Collect some nice fresh data
        it.set_description("Collecting data")
        key = collect_data(key, env, eps_policy, buffer, config.simulation_steps_per_epoch)

        dl = batched_dataloader(buffer, batch_size=config.batch_size, drop_last=True)

        it.set_description("Training")
        epoch_losses = []
        for batch in dl:
            s0 = jnp.asarray(batch["all_obs"])
            s1 = jnp.asarray(batch["all_next_obs"])
            a = jnp.asarray(batch["all_actions"])
            r = jnp.asarray(batch["reward"])
            d = jnp.asarray(batch["terminal"]).astype(np.float32)

            model, opt_state, loss_val = train_step(model, opt_state, target_model.network, s0, s1, a, r, d, config.gamma)
            epoch_losses.append(loss_val.item())

            target_model.update(model)

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        it.set_description(f"Loss = {avg_loss:.2f}")

        stats["loss"][epoch] = avg_loss
        wandb.log({"loss": avg_loss, "epoch": epoch})

        if epoch % config.save_every == 0:
            util.save_model(root_dir, model)

        if epoch % config.display_every == 0:
            q_policy = QPolicy(model)

            henv, _ = util.make_marl_env(config.env, config.env_kwargs | {"render_mode": "human"})

            obs_dict, _ = henv.reset(seed=config.seed + num_runs)
            obs_dict = util.value_map(obs_dict, obs_map)

            for _ in range(50):
                key, k1 = random.split(key)
                all_obs = jnp.concatenate([obs_dict[i][None] for i in agent_names])
                all_actions = q_policy.get_action(all_obs, k1)

                action_dict = {name: a.item() for name, a in zip(agent_names, all_actions)}

                obs_dict, rewards, terminated, truncated, _ = henv.step(action_dict)
                obs_dict = util.value_map(obs_dict, obs_map)

                if not henv.agents:
                    obs_dict, _ = henv.reset()
                    obs_dict = util.value_map(obs_dict, obs_map)
                    num_runs += 1

            henv.close()

        if epoch % config.eval_every == 0:
            q_policy = QPolicy(model)

            start = time.time()
            avg_reward, std_reward = evaluate.evaluate_multi_agent(config, seed=epoch, policy=q_policy, repeats=config.eval_reps, agent_names=agent_names)
            print(f"Evaluation completed. Took {time.time() - start:.3f}s")

            stats["avg_reward"][epoch] = avg_reward
            stats["std_reward"][epoch] = std_reward

            wandb.log({"reward": avg_reward, "epoch": epoch})

            # Save stats
            pickle.dump(stats, open(join(root_dir, "stats.pickle"), "wb"))

    # Save final model
    util.save_model(root_dir, model)

    env.close()

    # Save stats
    pickle.dump(stats, open(join(root_dir, "stats.pickle"), "wb"))

    util.plot_reward(stats)
    util.plot_loss(stats)

    wandb.finish()

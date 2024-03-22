# R255 Reinforcement Learning Project
TL;DR models are in `models` and some are included in the repo; the most relevant code
is probably `train.py` / `policy.py` / `network.py` and `multiagent_train.py` / `multiagent_network.py`
for the extension.

## Replicating Results
The models are included so you should be able to replicate everything.

### Lunar-lander
Table 1:
```
python evaluate.py -r lunar_lander/128_128_128 -n 50
# or '256_256', or '4096'
```
This will calculate one row of tab 1 and then start a pygame visualisation so you can see some
exciting lunar landing action. Oh yeah, it'll print the parallel score (which is biased
towards shorter runs, but much faster) and then the sequential score (which is what I
actually report).

Figure 2: Same as above; evaluate.py will plot the loss and reward graph for the given model.

### Multi-agent stuff
Table 2 and figure 4:
```
python evaluate.py -r [box_8_rot | box_16_fixed | box_16_rot | simple_spread]/[iql | vdn | qmix | pvdn] -n 20
```
Unfortunately `cooking2` is in the repo but will not work as I made some changes to
CookingZoo locally and didn't create a branch or put it inside this repo.

Example:
```
python evaluate.py -r box_16_rot/pvdn -n 1 --max_timestep 1000
```
`-n 1` because the calculation is very slow for box_env with 16 agents - this will just
show the agents jumping around with an extended max_timestep.



## Usage
`train.py`, `evaluate.py` and `multiagent_train.py` can all be run with argument `-r`
specifying the root directory of the model e.g.
```
python evaluate.py lunarlander/128d3
```
To view the box jump environment with random agent behaviour, just run
```
python box_env.py
```

## Codebase
```
train.py                    Single-agent training code (loss function etc)

# Stuff needed for DQN training
replay_buffer.py            Replay buffer
policy.py                   QPolicy and Epsilon policy
network.py                  DQN architecture (just an MLP) and the abstract QFunc class
target_network.py           EMA equinox target network
evaluate.py                 Track reward etc. When run as main, prints some stats.

# Multi-agent stuff
multiagent_train.py         Multi-agent training code
multiagent_network.py       IQL, VDN, QMIX, PairVDN implementations
box_env.py                  Box Jump environment

# Misc
config.py                   Config dataclass definition
util.py                     General utils
```

### Models
I use the following layout for models:
```
models/
 env_name/
  model_name/
   config.json
   model.eqx
   stats.pickle   # stores loss, reward, and Q values during training
```
where each `config.json` contains information corresponding to the `Config` object in `config.py`,
and replaces  command line arguments. As the models are pretty small, I've included a bunch in the repo, 
so you should be able to run evaluation and view behaviour as well as reward etc stats.

To start a new training run, create an empty directory somewhere in `models` with a `config.json`.
The options for `env` are theoretically any gym environment name for the single-agent case (though I only ever ran with
LunarLander), and one of `["knights_archers_zombies", "pursuit", "simple_spread", "boxjump"]`
in the multi-agent case (`cooking` and `cooking2` are unfortunately unavailable as I had to modify CookingZoo and
haven't created a branch).


## Setup
```
pip install swig
pip install -r requirements.txt
```
`requirements.txt` doesn't specify versions and has mainly very standard libraries.
I use `wandb` and haven't disabled it, so if running train use `WANDB_MODE=offline`.

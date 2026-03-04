# Bitcoin Under Volatile Block Rewards — MDP & A3C Tools

This repository contains the implementation of both the Markov Decision Process (MDP)-based tool and the Asynchronous Advantage Actor-Critic (A3C)-based tool introduced in the paper:

**"Bitcoin Under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining"**
Accepted at the ACM Conference on Computer and Communications Security (ACM CCS) 2025

## What this code does

`main.py` runs the A3C-based tool to analyze Bitcoin mining strategies under a **volatile block-reward model** in which rewards are dominated by **transaction fees**. It parameterizes an environment using **historical mempool statistics** for a specified period, modeling fee dynamics by (i) estimating a time–fee relationship and (ii) updating a mempool state that tracks transaction weight across sat/vByte fee ranges. It first prints baseline comparisons (e.g., honest-mining reward and profitability lower bounds) obtained through the MDP-based analysis and then launches **asynchronous multi-agent training/testing**: multiple worker agents interact with independent environment instances and asynchronously update a shared global **Actor–Critic** network to learn near-optimal adversarial policies (including **selfish mining** and **undercutting**, with undercutting treated as a **continuous-duration** action).   

## Run

Requirements:

* Python 3.9+
* PyTorch (CPU or CUDA)

From the repository root:
`python main.py` 


## Main files

* `main.py`: sets parameters, initializes environment-related quantities (including mempool/fee model parameters), prints baseline diagnostics (e.g., reward ratios and lower-bound profitability), and starts A3C training/testing with multiprocessing.
* `environment.py`: defines the blockchain mining environment (state, transitions, and rewards) used by training/testing workers.
* `model.py`: defines the `ActorCritic` network used by A3C.
* `train.py`: training worker process (A3C).
* `test.py`: testing/evaluation worker process (A3C).
* `my_optim.py`: shared optimizer utilities for multiprocessing A3C.
* `lower_bound_profits2.py`: implements the MDP-based lower-bound profitability analysis introduced in the paper.


## Configuration (main.py)

All experiment settings are defined in `main.py` in three dictionaries:

* `A3C_args`: A3C hyperparameters and multiprocessing settings
* `BTC_args`: blockchain/mining environment parameters
* `Mempool_args`: mempool statistics and fee-model parameters

### A3C_args

* `lr`: learning rate
* `gamma`: discount factor
* `gae_lambda`: GAE parameter
* `entropy_coef`: entropy regularization coefficient
* `value_loss_coef`: value loss coefficient
* `max_grad_norm`: gradient clipping threshold
* `num_processes_train`: number of training workers
* `num_processes_test`: number of testing workers
* `num_steps`: rollout length per worker update
* `max_episode_length`: maximum training episode length
* `testing_episode_length`: evaluation episode length
* `seed`: random seed
* `no_shared`: if `True`, disables the shared optimizer

### BTC_args

* `adversarial_ratio`: adversary mining fraction
* `rational_ratio`: rational miner fraction
* `connectivity`: connectivity parameter used by the environment
* `max_fork`: maximum fork length tracked in the state (also affects state size)
* `mining_time`: mining time parameter used by the environment
* `block_fixed_reward`: fixed reward component (set to `0` in the default config shown in `main.py`)
* `attack_type`: attack mode selector (`0`: selfish_mining_undercutting, `1`: selfish_mining, `2`: undercutting)
* `noise`: enables/disables noise in the environment rewarding mechanism (`False` by default)
* `epsilon`: incentivizing parameter used by the environment (`0` by default)
* `Diff_adjusted`: difficulty-adjustment flag (`False` before adjustment, `True` after)

### Mempool_args

`Mempool_args` defines the mempool time window and discretization used by the fee/reward model. `main.py` includes multiple pre-defined `Mempool_args` blocks (commented/uncommented) for different historical ranges; switching between them changes the mempool regime used by the environment.


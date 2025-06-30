import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from math import pi
from torch import from_numpy, normal, clamp
# from envs import create_atari_env
from environment import Bitcoin_Transaction_Fee_Model
from model import ActorCritic
from multiprocessing import Pool, current_process
from Possible_actions import selfish_mining_undercutting, selfish_mining, undercutting
import math

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def normalProb(x, mu, sigma, pi, ONE, TWO):
    a = x.sub(mu).square().neg().div(sigma.mul(TWO)).exp()
    b = ONE.div(sigma.mul(pi.mul(TWO)).sqrt())
    return a.mul(b)

def train(rank, A3C_args, BTC_args, Mempool_args, shared_model_a, optimizer_a, counter, profit_ratio_max, lock):
    torch.manual_seed(A3C_args.seed + rank)
    print("Training agent rank:", rank, current_process().name, flush=True)
    possible_actions = [selfish_mining_undercutting, selfish_mining, undercutting]
    env = Bitcoin_Transaction_Fee_Model(BTC_args, Mempool_args)
    model_a = ActorCritic(A3C_args, BTC_args)
    if optimizer_a is None:
        optimizer_a = optim.Adam(shared_model_a.parameters(), lr=A3C_args.lr)
    model_a.train()
    normal_adversarial_block_reward_per_min = Mempool_args.normal_adversarial_block_reward_per_min
    normal_adversarial_block_reward = BTC_args.mining_time * normal_adversarial_block_reward_per_min / BTC_args.adversarial_ratio
    state = env.reset()
    done = False
    cx_a = torch.zeros(1, A3C_args.N_nodes_per_layer)
    hx_a = torch.zeros(1, A3C_args.N_nodes_per_layer)
    episode_length = 0
    while True:
        model_a.load_state_dict(shared_model_a.state_dict())
        if done:
            state = env.reset()
            cx_a = torch.zeros(1, A3C_args.N_nodes_per_layer)
            hx_a = torch.zeros(1, A3C_args.N_nodes_per_layer)
        else:
            cx_a = cx_a.detach()
            hx_a = hx_a.detach()
        values_a = []
        values_diff = []
        log_probs_a = []
        rewards_a = []
        difficulties = []
        entropies_a = []

        for step in range(A3C_args.num_steps):
            episode_length += 1
            possible_action_set = possible_actions[BTC_args.attack_type](state, env)
            possible_action_set = torch.tensor(possible_action_set, dtype=torch.float)
            state = torch.tensor(state, dtype=torch.float)
            
            value_a, value_diff, q_a, mu, (hx_a, cx_a) = model_a((state.unsqueeze(0),(hx_a, cx_a)))
            
            noise_var = torch.tensor(A3C_args.noise_std**2.0).float()
            noise = normal(torch.tensor(0.0).float(), torch.tensor(A3C_args.noise_std).float())
            action_time = mu.add(noise).data
            prob_undercut = normalProb(action_time, mu, noise_var, torch.tensor(pi).float(), torch.ones(()).float(), torch.tensor(2.0).float())
            action_time = clamp(action_time, 0, 1)
            entropy_undercut = (
                noise_var.mul((torch.tensor(2.0).float()).mul(torch.tensor(pi).float()))
                .log()
                .add(torch.ones(()).float())
                .mul(torch.tensor(0.5).float())
            )
            EPS = torch.tensor(1e-6).float()
            log_prob_undercut = prob_undercut.add(EPS).log()
            
            q_total = torch.where(possible_action_set == -1e20, -1e20, q_a)

            if torch.all(q_total[0] == -1e20):
                print("No action is possible to take (potential error in the code)", state[:2], q_a, possible_action_set, flush=True)
            else:
                prob_total = F.softmax(q_total, dim=-1)
                log_prob_total = F.log_softmax(q_total, dim=-1)
                entropy_total = -(log_prob_total * prob_total).sum(1, keepdim=True)
                entropy_total = entropy_total.add(entropy_undercut.mul(prob_total[0][4 + env.len_abort]))
                entropies_a.append(entropy_total)
                action = prob_total.multinomial(num_samples=1).detach()

            log_prob = log_prob_total.gather(1, action)
            if action.numpy() == 4 + env.len_abort:
                log_prob = log_prob.add(log_prob_undercut)
            
            if rank == 1 and episode_length % A3C_args.training_print_length == 0:
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', flush=True)
                print('value_diff', value_diff.detach() /  A3C_args.reward_scaling_coefficient * (1 - A3C_args.gamma), flush=True)
                print('rank, episode_length', rank, episode_length, flush=True)
                print('state:', state[:2], flush=True)
                print('q_a', q_a, flush=True)
                print('possible action', possible_action_set, flush=True)
                print('prob_total', prob_total, flush=True)
                print('action_time', 80 * (action_time), flush=True)
                print('mu', mu, flush=True)
                print('entropy_total', entropy_total, flush=True)
                print('entropy_undercut', entropy_undercut, flush=True)
                print('memPool Canonical', env.current_state[5 + 2 * env.max_fork: 5 + 2 * env.max_fork + env.N_memPool_section], flush=True)
                print('log prob', log_prob, flush=True)
                print('log prob undercut', log_prob_undercut, flush=True)
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', flush=True)

            Diff_estimate = value_diff.detach() /  A3C_args.reward_scaling_coefficient * (1 - A3C_args.gamma)
            Diff_estimate = min(max(Diff_estimate, 1 - BTC_args.adversarial_ratio), 1)
            if BTC_args.Diff_adjusted:
                state, n_a, n_h, adversary_reward, time_block, diff_pre_DAM = env.step(action.numpy(), action_time.numpy(), Diff_estimate)
            else: 
                state, n_a, n_h, adversary_reward, time_block, diff_pre_DAM = env.step(action.numpy(), action_time.numpy())
            normalized_reward_a = adversary_reward / normal_adversarial_block_reward
            normalized_cost = profit_ratio_max.value * (n_a + n_h) if BTC_args.Diff_adjusted else BTC_args.adversarial_ratio
#             normalized_cost = profit_ratio_max.value * (n_a + n_h) if BTC_args.Diff_adjusted else 0.05
            rewards_a.append((normalized_reward_a - normalized_cost) * A3C_args.reward_scaling_coefficient)
            
            diff = n_a + n_h if BTC_args.Diff_adjusted else 1
            difficulties.append(diff * A3C_args.reward_scaling_coefficient)
            
            values_a.append(value_a)
            values_diff.append(value_diff)
            log_probs_a.append(log_prob)
            
            with lock:
                counter.value += 1
            done = episode_length >= A3C_args.max_episode_length 
            if done:
                if rank == 1:
                    print(episode_length)
                    print(value_a)
                episode_length = 0
                state = env.reset()
                break

        R_a = torch.zeros(1, 1)
        if not done:
            state_final = torch.tensor(state, dtype=torch.float)
            value_a, value_diff, _, _, _ = model_a((state_final.unsqueeze(0), (hx_a, cx_a)))
            R_a = value_a.detach()
            R_diff = value_diff.detach()
        values_a.append(R_a)
        values_diff.append(R_diff)
        policy_loss_a = 0
        value_loss_a = 0
        value_loss_diff = 0
        gae_a = torch.zeros(1, 1)
        for i in reversed(range(len(rewards_a))):
            R_a = A3C_args.gamma * R_a + rewards_a[i]
            R_diff = A3C_args.gamma * R_diff + difficulties[i]
            advantage_a = R_a - values_a[i]
            advantage_diff = R_diff - values_diff[i]
            value_loss_a = value_loss_a + 0.5 * advantage_a.pow(2)
            value_loss_diff = value_loss_diff + 0.5 * advantage_diff.pow(2)
            # Generalized Advantage Estimation
            delta_t_a = rewards_a[i] + A3C_args.gamma * values_a[i + 1] - values_a[i]
            gae_a = gae_a * A3C_args.gamma * A3C_args.gae_lambda + delta_t_a
            policy_loss_a = policy_loss_a - log_probs_a[i] * gae_a.detach() - A3C_args.entropy_coef * entropies_a[i]
        
        optimizer_a.zero_grad()
        (policy_loss_a + A3C_args.value_loss_coef * (value_loss_a + value_loss_diff)).backward()
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), A3C_args.max_grad_norm)
        ensure_shared_grads(model_a, shared_model_a)
        optimizer_a.step()
        
        if rank == 1 and episode_length % A3C_args.training_print_length == 0:
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', flush=True)
            print('policy_loss_a', policy_loss_a, flush=True)
            print('value_loss_a', value_loss_a, flush=True)
            print('total loss:', policy_loss_a + A3C_args.value_loss_coef * value_loss_a, flush=True)
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', flush=True)

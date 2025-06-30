import time
from environment import Bitcoin_Transaction_Fee_Model
import numpy as np
import torch
import torch.nn.functional as F
import os
from model import ActorCritic
from multiprocessing import Pool, current_process
from Possible_actions import selfish_mining_undercutting, selfish_mining, undercutting

def save_checkpoint(network, name, BTC_args, Mempool_args):
        print('... saving checkpoint ...', flush=True)
        checkpoint_dir = 'A3C_results' + '_' + '2' + '_' + 'mining_share' + '_' + str(BTC_args.adversarial_ratio) + '_' + 'connectivity' + '_' + str(BTC_args.connectivity) \
                         + '_' + 'rationality' + '_' + str(BTC_args.rational_ratio) + '_' + 'from' + '_' + Mempool_args.start_date_str + '_' + 'to' + '_' + Mempool_args.end_date_str + '_' + 'nSection' + '_' \
                         + str(Mempool_args.N_memPool_section) + '_' + 'lRange' + '_' + str(Mempool_args.sat_per_byte_range_length)
        checkpoint_dir = checkpoint_dir.replace("/", "-").replace(":", "-")
        checkpoint_file = os.path.join(checkpoint_dir, name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(network.state_dict(), checkpoint_file)

def load_checkpoint(network, name, BTC_args, Mempool_args):
        print('... loading checkpoint ...')
        checkpoint_dir = 'A3C_results' + '_' + '2' + '_' + 'mining_share' + '_' + str(BTC_args.adversarial_ratio) + '_' + 'connectivity' + '_' + str(BTC_args.connectivity) \
                         + '_' + 'rationality' + '_' + str(BTC_args.rational_ratio) + '_' + 'from' + '_' + Mempool_args.start_date_str + '_' + 'to' + '_' + Mempool_args.end_date_str + '_' + 'nSection' + '_' \
                         + str(Mempool_args.N_memPool_section) + '_' + 'lRange' + '_' + str(Mempool_args.sat_per_byte_range_length)
        checkpoint_dir = checkpoint_dir.replace("/", "-").replace(":", "-")
        checkpoint_file = os.path.join(checkpoint_dir, name)
        network.load_state_dict(torch.load(checkpoint_file))


def test(rank, A3C_args, BTC_args, Mempool_args, shared_model_a, counter, profit_ratio_max):
    print("Testing agent rank:", rank, current_process().name, flush=True)
    torch.manual_seed(A3C_args.seed + rank)
    possible_actions = [selfish_mining_undercutting, selfish_mining, undercutting]
    env = Bitcoin_Transaction_Fee_Model(BTC_args, Mempool_args)
    model_a = ActorCritic(A3C_args, BTC_args)
    model_a.eval()
    state = env.reset()
    reward_sum_a = 0
    time_passage = 0
    n_total = 0
    diff_total = 1e-7
    start_time = time.time()
    episode_length = 0
    if A3C_args.long_range_testing:
        name = 'rank' + str(rank) + '_' + 'index' + str(A3C_args.testing_index)
        load_checkpoint(model_a, name, BTC_args, Mempool_args)
    else:
        model_a.load_state_dict(shared_model_a.state_dict())
    cx_a = torch.zeros(1, A3C_args.N_nodes_per_layer)
    hx_a = torch.zeros(1, A3C_args.N_nodes_per_layer)
    profits = np.zeros(3)
    while True:
        episode_length += 1
        possible_action_set = possible_actions[BTC_args.attack_type](state, env)
        possible_action_set = torch.tensor(possible_action_set, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)
        cx_a = cx_a.detach()
        hx_a = hx_a.detach()
        with torch.no_grad():
            value_a, value_diff, q_a, mu, (hx_a, cx_a) = model_a((state.unsqueeze(0), (hx_a, cx_a)))
        q_total = torch.where(possible_action_set == -1e20, -1e20, q_a)

        if torch.all(q_total[0] == -1e20):
            print("No action is possible to take (potential error in the code)", state[:2], q_a, possible_action_set, flush=True)
        else:
            prob_total = F.softmax(q_total, dim=-1)
            action = prob_total.max(1, keepdim=True)[1].numpy()
        if BTC_args.Diff_adjusted:
            state, n_a, n_h, adversary_reward, duration, diff_a_h = env.step(action[0, 0], mu.numpy(), min(max(n_total/diff_total, 1 - BTC_args.adversarial_ratio), 1))
        else:
            state, n_a, n_h, adversary_reward, duration, diff_a_h = env.step(action[0, 0], mu.numpy())
        reward_sum_a += adversary_reward
        time_passage += duration
        diff_total += diff_a_h
        n_total += n_a + n_h

        if episode_length % A3C_args.testing_episode_length == 0:
            if BTC_args.Diff_adjusted:
                profit = (reward_sum_a/time_passage/Mempool_args.normal_adversarial_block_reward_per_min)
                if not A3C_args.long_range_testing and profit_ratio_max.value < profit:
                    profit_ratio_max.value = profit
            else: 
                profit = reward_sum_a/time_passage/Mempool_args.normal_adversarial_block_reward_per_min
            print("**************************", " TEST RESULT ", "**************************", flush=True)
            print("Time {}, num steps {}, FPS {:.0f}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time), episode_length), flush=True)
            print("Episode reward:", profit)
            if not A3C_args.long_range_testing and profit > min(profits):
                profits[np.argmin(profits)] = profit
                name = 'rank' + str(rank) + '_' + 'index' + str(np.where(profits == profit)[0][0])
                save_checkpoint(model_a, name, BTC_args, Mempool_args)
            if not A3C_args.long_range_testing:
                model_a.load_state_dict(shared_model_a.state_dict())
                print('Max profit so far:', max(profits), flush=True)
            print("**************************", "*************", "**************************", flush=True)

            reward_sum_a = 0
            time_passage = 0
            diff_total = 1e-7
            n_total = 0

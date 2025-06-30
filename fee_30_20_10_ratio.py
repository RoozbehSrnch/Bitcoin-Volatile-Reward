from environment import Bitcoin_Transaction_Fee_Model
import numpy as np

def choose_action_honest(state):
    if state[0] == 1:
        action = 0
    else:
        if state[0] != 0 or state[1] != 1:
            raise Exception("This is not possible.")
        action = 3
    return action

def fee_ratio(BTC_args, Mempool_args):
    print('Obtaining the average fee ratio of 30-minute and 20-minute blocks to a 10-minute block ...', flush=True)
    mining_time = BTC_args.mining_time
    adversarial_ratio = BTC_args.adversarial_ratio
    BTC_args.mining_time = 10
    BTC_args.adversarial_ratio = 1
    env = Bitcoin_Transaction_Fee_Model(BTC_args, Mempool_args)
    state = env.reset()
    cnt = 0
    ten_block_reward = 0
    while cnt <= Mempool_args.N_steps_honest_mining:
        action = choose_action_honest(state)
        state_, n_a, n_h, adversary_reward, duration,_ = env.step(action, 0)
        ten_block_reward += adversary_reward
        state = state_
        cnt += 1
#         if cnt % 50000 == 0:
#             print("Number of steps:", cnt, flush=True)
            
    print("Average 10-min block fee:", ten_block_reward/cnt, flush=True)
    BTC_args.mining_time = 20
    BTC_args.adversarial_ratio = 1
    env = Bitcoin_Transaction_Fee_Model(BTC_args, Mempool_args)
    state = env.reset()
    cnt = 0
    twenty_block_reward = 0
    while cnt <= Mempool_args.N_steps_honest_mining:
        action = choose_action_honest(state)
        state_, n_a, n_h, adversary_reward, duration,_ = env.step(action, 0)
        twenty_block_reward += adversary_reward
        state = state_
        cnt += 1
#         if cnt % 50000 == 0:
#             print("Number of steps:", cnt, flush=True)
            
    print("Average 20-min block fee:", twenty_block_reward/cnt, flush=True)
    BTC_args.mining_time = 30
    BTC_args.adversarial_ratio = 1
    env = Bitcoin_Transaction_Fee_Model(BTC_args, Mempool_args)
    state = env.reset()
    cnt = 0
    thirty_block_reward = 0
    while cnt <= Mempool_args.N_steps_honest_mining:
        action = choose_action_honest(state)
        state_, n_a, n_h, adversary_reward, duration,_ = env.step(action, 0)
        thirty_block_reward += adversary_reward
        state = state_
        cnt += 1
#         if cnt % 50000 == 0:
#             print("Number of steps:", cnt, flush=True)
            
    print("Average 30-min block fee:", thirty_block_reward/cnt, flush=True)
    ratio_20_10 = twenty_block_reward / ten_block_reward
    ratio_30_10 = thirty_block_reward / ten_block_reward
    print('Average ratio of a 20-min block to a 10-min block:', ratio_20_10, flush=True)
    print('Average ratio of a 30-min block to a 10-min block:', ratio_30_10, flush=True)
    BTC_args.mining_time = mining_time
    BTC_args.adversarial_ratio = adversarial_ratio
    return ratio_20_10, ratio_30_10
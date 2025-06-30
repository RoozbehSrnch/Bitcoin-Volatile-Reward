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

def Honest_mining_block_reward(BTC_args, Mempool_args):
    print('Obtaining the normal adversarial block reward per minute while mining honestly ...', flush=True)
    env = Bitcoin_Transaction_Fee_Model(BTC_args, Mempool_args)
    state = env.reset()
    memPool_avg = np.zeros(env.N_memPool_section)
    cnt = 0
    normal_block_reward = 0
    t = 0
    while cnt <= Mempool_args.N_steps_honest_mining:
        action = choose_action_honest(state)
        state_, n_a, n_h, adversary_reward, duration,_ = env.step(action, 0)
        memPool = env.current_state[5 + 2 * env.max_fork: 5 + 2 * env.max_fork + env.N_memPool_section]
        memPool_avg = memPool_avg + memPool
        normal_block_reward += adversary_reward
        state = state_
        cnt += 1
        t += duration
#         if cnt % 50000 == 0:
#             print("Number of steps:", cnt, flush=True)
    normal_adversarial_block_reward_per_min = normal_block_reward/t
    # memPool_avg = memPool_avg / cnt
    # print('memPool_avg', memPool_avg, flush=True)
    print('Normal adversarial block reward per minute', normal_adversarial_block_reward_per_min, flush=True)
    return normal_adversarial_block_reward_per_min
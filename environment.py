import numpy as np
import math

class Bitcoin_Transaction_Fee_Model():
    def __init__(self, BTC_args, Mempool_args):
        self.adversarial = BTC_args.adversarial_ratio
        self.honest = 1 - self.adversarial
        self.rational = BTC_args.rational_ratio
        self.compliant = 1 - BTC_args.rational_ratio
        self.connectivity = BTC_args.connectivity
        self.max_fork = BTC_args.max_fork
        self.len_abort = BTC_args.len_abort
        self.mining_time = BTC_args.mining_time
        self.state_length = BTC_args.state_length
        self.block_fixed_reward = BTC_args.block_fixed_reward
        self.max_block_size = BTC_args.max_block_size
        self.noise = BTC_args.noise
        self.epsilon = BTC_args.epsilon

        self.N_memPool_section = Mempool_args.N_memPool_section
        self.initial_memPool = np.zeros(self.N_memPool_section)
        self.memPool_max_size = np.full(self.N_memPool_section, 10 * self.max_block_size)
        self.memPool_min_size = np.array([self.max_block_size] + [0] * (self.N_memPool_section - 1))
        self.memPool_coefficient = Mempool_args.coefficient_mempool_reward
        self.memPool_noise_std = Mempool_args.noise_std_mempool_reward
        self.memPool_avg_rewards = 0.01 * np.arange(Mempool_args.Base_sat_per_byte_range + Mempool_args.sat_per_byte_range_length/2, Mempool_args.Base_sat_per_byte_range + Mempool_args.N_memPool_section * Mempool_args.sat_per_byte_range_length, Mempool_args.sat_per_byte_range_length)

        
    def reset(self):
        a = 0
        h = 0
        match = 0
        latest = 0
        undercut = 0
        rewards_a = np.zeros(self.max_fork)
        rewards_h = np.zeros(self.max_fork)
        memPool_canonical = self.initial_memPool
        memPools_a = np.zeros((self.max_fork, self.N_memPool_section))
        memPools_h = np.zeros((self.max_fork, self.N_memPool_section))
        self.current_state = np.zeros(self.state_length)
        
        adversary_mine, _, time = self.mine(self.current_state, False)
        memPools_a, memPools_h, memPool_canonical = self.update_memPool_time_passage(memPools_a, memPools_h,
                                                                                     memPool_canonical, a, h,
                                                                                     time)

        if adversary_mine:
            memPools_a, rewards_a[0] = self.update_memPool_new_block(memPools_a, memPool_canonical, 0)
            a = 1
        else:
            memPools_h, rewards_h[0] = self.update_memPool_new_block(memPools_h, memPool_canonical, 0)
            h = 1
            latest = 1

        state_ = np.concatenate(([a, h, match, latest, undercut], rewards_a, rewards_h, memPool_canonical), 0)
        for i in range(self.max_fork):
            state_ = np.concatenate((state_, memPools_a[i]), 0)
        for i in range(self.max_fork):
            state_ = np.concatenate((state_, memPools_h[i]), 0)
        self.current_state = state_
        return self.current_state

    def mine(self, state, match_active, current_diff = 1):
        adversary_mine = False
        match_win = False
        state = np.array(state)
        a = int(state[0])
        h = int(state[1])
        match = state[2]
        latest = state[3]
        undercut = state[4]
        rewards_a = state[5: 5 + self.max_fork]
        rewards_h = state[5 + self.max_fork: 5 + 2 * self.max_fork]
        memPool_canonical = state[5 + 2 * self.max_fork: 5 + 2 * self.max_fork + self.N_memPool_section]
        state = state[5 + 2 * self.max_fork + self.N_memPool_section:]
        memPools_a = np.zeros((self.max_fork, self.N_memPool_section))
        for i in range(self.max_fork):
            memPools_a[i] = state[i * self.N_memPool_section: (i + 1) * self.N_memPool_section]

        state = state[self.max_fork * self.N_memPool_section:]
        memPools_h = np.zeros((self.max_fork, self.N_memPool_section))
        for i in range(self.max_fork):
            memPools_h[i] = state[i * self.N_memPool_section: (i + 1) * self.N_memPool_section]

        avg_time_adversary = self.mining_time * current_diff / self.adversarial if self.adversarial > 0 else 1e9
        avg_time_honest = self.mining_time * current_diff / self.honest if self.honest > 0 else 1e9
        time_adversary = np.random.exponential(avg_time_adversary)
        time_honest = np.random.exponential(avg_time_honest)
        if time_adversary <= time_honest:
            adversary_mine = True
            time = time_adversary
        else:
            time = time_honest
            # if match_active and (np.random.rand() <= self.rational or np.random.rand() <= self.connectivity):
            #     match_win = True
            if match_active and undercut == 1:
                H_miner = np.random.rand()
                if H_miner <= self.rational:
                    _, block_reward_h_fork = self.update_memPool_new_block(memPools_h, memPool_canonical, h)
                    _, block_reward_a_fork = self.update_memPool_new_block(memPools_a, memPool_canonical, h)
                    if block_reward_a_fork >= block_reward_h_fork + self.epsilon:
                        match_win = True
                    else:
                        match_win = False   
            elif match_active and undercut == 0:
                H_miner = np.random.rand()
                if H_miner <= self.rational:
                    _, block_reward_h_fork = self.update_memPool_new_block(memPools_h, memPool_canonical, h)
                    _, block_reward_a_fork = self.update_memPool_new_block(memPools_a, memPool_canonical, h)
                    if block_reward_a_fork >= block_reward_h_fork + self.epsilon:
                        match_win = True
                    elif block_reward_h_fork > block_reward_a_fork + self.epsilon:
                        match_win = False
                    else:
                        if np.random.rand() <= self.connectivity:
                            match_win = True
                elif self.rational < H_miner <= self.rational + (1-self.rational)*self.connectivity:
                    match_win = True

        return adversary_mine, match_win, time

    def step(self, action, action_time, current_diff = 1):
        self.previous_state = self.current_state.copy()
        action_time = 80 * action_time
        if action == 1:
            match_active = True
        else:
            match_active = False

        adversary_mine, match_win, time = self.mine(self.previous_state, match_active, current_diff)
        # print(time)
        self.current_state, n_adversary_block, adversary_reward, n_honest_block, honest_reward, difficulty = \
            self.next_state(self.previous_state, action, action_time, adversary_mine, match_win, time)
        # if action ==3:
        # print('environmeeent9999999999999999999999999999', time, action, n_adversary_block, adversary_reward, n_honest_block, honest_reward)
        return self.current_state, n_adversary_block, n_honest_block, adversary_reward, time, difficulty


    def total_reward(self, rewards):
        reward = len(rewards) * self.block_fixed_reward + np.sum(rewards)
        return reward

    def volume_new_transactions(self, time, noise):
        added_volume = np.zeros(self.N_memPool_section)
        for index in range(self.N_memPool_section):
            added_volume[index] = self.memPool_coefficient[index][0] * (time ** self.memPool_coefficient[index][1]) + self.memPool_coefficient[index][2]
            added_volume[index] = max(0, added_volume[index] + noise[index])
        # print('8888888888', added_volume)
        return added_volume

    def update_memPool_time_passage(self, memPools_a, memPools_h, memPool_canonical, a, h, time):
        noise = np.zeros(self.N_memPool_section)
        if self.noise:
            for index in range(self.N_memPool_section):
                noise[index] = np.random.normal(0, self.memPool_noise_std[index])
        for i in range(a):
            memPools_a[i] = np.maximum(np.minimum(self.volume_new_transactions(time, noise) + memPools_a[i], self.memPool_max_size), self.memPool_min_size)
        for i in range(h):
            memPools_h[i] = np.maximum(np.minimum(self.volume_new_transactions(time, noise) + memPools_h[i], self.memPool_max_size), self.memPool_min_size)
        # print(len(self.memPool_rates))
        # print(len(memPool_canonical))
        memPool_canonical = np.maximum(np.minimum(self.volume_new_transactions(time, noise) + memPool_canonical, self.memPool_max_size), self.memPool_min_size)
        # print('999999999999999999999999999999999999999999999999999999', memPool_canonical)
        return memPools_a, memPools_h, memPool_canonical

    def update_memPool_new_block(self, memPools, memPool_canonical, height, max_block_size=None):
        if max_block_size is None:
            max_block_size = self.max_block_size
        # else:
        #     print('$$$$$$$$$$$$$$', max_block_size)
        block_reward = 0
        block_size = 0
        if height == 0:
            memPool = memPool_canonical.copy()
        else:
            memPool = memPools[height - 1].copy()
        # print('memPool_1', memPool, height)

        for i in reversed(range(self.N_memPool_section)):
            if memPool[i] >= max_block_size - block_size:
                block_reward = block_reward + (max_block_size - block_size) * self.memPool_avg_rewards[i]
                memPool[i] = memPool[i] - (max_block_size - block_size)
                block_size = max_block_size
                break
            else:
                block_reward = block_reward + memPool[i] * self.memPool_avg_rewards[i]
                block_size = block_size + memPool[i]
                memPool[i] = 0
        memPools[height] = memPool
        # print('memPool_2', memPools[height], height)
        return memPools, block_reward

    def update_memPool_undercut(self, memPools_a, memPools_h, memPool_canonical, h, rewards_h, eps):
        if eps == 0 and h == 1:
            return memPools_h, rewards_h[0]
        _, block_reward_h_fork = self.update_memPool_new_block(memPools_h, memPool_canonical, h)
        potential_memPools_a, potential_block_reward = self.update_memPool_new_block(memPools_a, memPool_canonical, h-1)
        _, block_reward_a_fork = self.update_memPool_new_block(potential_memPools_a, memPool_canonical, h)
        if block_reward_a_fork >= block_reward_h_fork + eps:
            return potential_memPools_a, potential_block_reward
        max_block_size = self.max_block_size
        min_block_size = 0
        block_size = (max_block_size + min_block_size) / 2
        # print(block_reward_h_fork)
        while True:
            
            potential_memPools_a, potential_block_reward = self.update_memPool_new_block(memPools_a, memPool_canonical,
                                                                                         h - 1, max_block_size=block_size)
            _, block_reward_a_fork = self.update_memPool_new_block(potential_memPools_a, memPool_canonical, h)
            # print(block_size, block_reward_a_fork)
            if block_reward_a_fork < block_reward_h_fork + eps:
                max_block_size = block_size
            elif block_reward_a_fork > block_reward_h_fork + eps + 0.01:
                min_block_size = block_size
            else:
                break
            block_size = (max_block_size + min_block_size) / 2
        return potential_memPools_a, potential_block_reward

    def next_state(self, state_, action, action_time, adversary_mine, match_win, time):
        # state format: (L_a, L_h, match, latest, undercut, rewards_a, rewards_h, memPools_a, memPools_h)
        # state = np.array([int(state[i]) for i in range(len(state))])
        state = np.array(state_)
        a = int(state[0])
        h = int(state[1])
        match = state[2]
        latest = state[3]
        undercut = state[4]
        rewards_a = state[5: 5 + self.max_fork]
        rewards_h = state[5 + self.max_fork: 5 + 2 * self.max_fork]
        memPool_canonical = state[5 + 2 * self.max_fork: 5 + 2 * self.max_fork + self.N_memPool_section]

        state = state[5 + 2*self.max_fork + self.N_memPool_section:]
        memPools_a = np.zeros((self.max_fork, self.N_memPool_section))
        for i in range(self.max_fork):
            memPools_a[i] = state[i * self.N_memPool_section: (i+1) * self.N_memPool_section]

        state = state[self.max_fork * self.N_memPool_section:]
        memPools_h = np.zeros((self.max_fork, self.N_memPool_section))
        for i in range(self.max_fork):
            memPools_h[i] = state[i * self.N_memPool_section: (i+1) * self.N_memPool_section]
        # print('#####################################')
        # print('memPool_before', memPools_a[0], 'time', time, 'action', action)
        memPools_a, memPools_h, memPool_canonical = self.update_memPool_time_passage(memPools_a, memPools_h,
                                                                                     memPool_canonical, a, h,
                                                                                     time)
        # print('memPool_after', memPools_a[0], 'time', time, 'action', action)
        if action == 4 + self.len_abort and time > action_time:
            action = 3  # abort
            # print('jump to 3')
        d = 0
        # override
        if action == 0:
            # When match = 1, override is feasible.
            match = 0
            if adversary_mine == 1:
                N_a = h + 1
                R_a = self.total_reward(rewards_a[:h + 1])
                N_h = 0
                R_h = 0
                d = 2*h + 1
                # print('memPool_canonical_before', memPool_canonical)
                memPool_canonical = memPools_a[h].copy()
                # print('memPool_canonical_after', memPool_canonical)
                memPools_h = np.zeros((self.max_fork, self.N_memPool_section))
                memPools_a = np.roll(memPools_a, -(h+1), axis=0)
                memPools_a[a-h-1:, ] = 0
                # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                memPools_a, block_reward = self.update_memPool_new_block(memPools_a, memPool_canonical, a - h - 1)
                
                # print('memPool_after2', memPools_a[0], 'time', time, 'action', action)
                # print('#####################################')

                rewards_a = np.roll(rewards_a, -(h+1))
                rewards_a[a - h - 1] = block_reward
                rewards_a[a - h:] = 0
                rewards_h = np.zeros(self.max_fork)
                a = a - h
                h = 0
                latest = 0
            else:
                N_a = h + 1
                R_a = self.total_reward(rewards_a[:h + 1])
                N_h = 0
                R_h = 0
                d = 2*h + 1
                memPool_canonical = memPools_a[h].copy()
                memPools_h = np.zeros((self.max_fork, self.N_memPool_section))
                memPools_h, block_reward = self.update_memPool_new_block(memPools_h, memPool_canonical, 0)
                memPools_a = np.roll(memPools_a, -(h + 1), axis=0)
                memPools_a[a - h - 1:, ] = 0

                rewards_a = np.roll(rewards_a, -(h + 1))
                rewards_a[a - h - 1:] = 0
                rewards_h[0] = block_reward
                rewards_h[1:] = 0
                a = a - h - 1
                h = 1
                latest = 1
            
        # match
        elif action == 1:
            undercut = 0
            if adversary_mine == 1:
                N_a = 0
                R_a = 0
                N_h = 0
                R_h = 0

                memPools_a, block_reward = self.update_memPool_new_block(memPools_a, memPool_canonical, a)

                rewards_a[a] = block_reward
                a = a + 1
                match = 1
                latest = 0
            elif adversary_mine == 0 and match_win == 1:
                N_a = h
                R_a = self.total_reward(rewards_a[:h])
                N_h = 0
                R_h = 0
                d = 2*h
                memPool_canonical = memPools_a[h-1].copy()
                memPools_h = np.zeros((self.max_fork, self.N_memPool_section))
                memPools_h, block_reward = self.update_memPool_new_block(memPools_h, memPool_canonical, 0)
                memPools_a = np.roll(memPools_a, -h, axis=0)
                memPools_a[a - h:, ] = 0

                rewards_a = np.roll(rewards_a, -h)
                rewards_a[a - h:] = 0
                rewards_h[0] = block_reward
                rewards_h[1:] = 0
                a = a-h
                h = 1
                # match = 0
                latest = 1
            elif adversary_mine == 0 and match_win == 0:
                N_a = 0
                R_a = 0
                N_h = 0
                R_h = 0

                memPools_h, block_reward = self.update_memPool_new_block(memPools_h, memPool_canonical, h)

                rewards_h[h] = block_reward
                h = h+1
                # match = 0
                latest = 1

        # wait
        elif action == 2:
            if match != 1:
                if adversary_mine == 1:
                    N_a = 0
                    R_a = 0
                    N_h = 0
                    R_h = 0

                    memPools_a, block_reward = self.update_memPool_new_block(memPools_a, memPool_canonical, a)

                    rewards_a[a] = block_reward
                    a = a + 1
                    latest = 0
                else:
                    N_a = 0
                    R_a = 0
                    N_h = 0
                    R_h = 0

                    memPools_h, block_reward = self.update_memPool_new_block(memPools_h, memPool_canonical, h)

                    rewards_h[h] = block_reward
                    h = h + 1
                    latest = 1
            elif match == 1:
                if adversary_mine == 1:
                    N_a = 0
                    R_a = 0
                    N_h = 0
                    R_h = 0

                    memPools_a, block_reward = self.update_memPool_new_block(memPools_a, memPool_canonical, a)

                    rewards_a[a] = block_reward
                    a = a + 1
                    latest = 0
                elif adversary_mine == 0 and match_win == 1:
                    N_a = h
                    R_a = self.total_reward(rewards_a[:h])
                    N_h = 0
                    R_h = 0
                    d = 2*h
                    memPool_canonical = memPools_a[h - 1].copy()
                    memPools_h = np.zeros((self.max_fork, self.N_memPool_section))
                    memPools_h, block_reward = self.update_memPool_new_block(memPools_h, memPool_canonical, 0)
                    memPools_a = np.roll(memPools_a, -h, axis=0)
                    memPools_a[a - h:, ] = 0

                    rewards_a = np.roll(rewards_a, -h)
                    rewards_a[a - h:] = 0
                    rewards_h[0] = block_reward
                    rewards_h[1:] = 0
                    a = a - h
                    h = 1
                    match = 0
                    latest = 1
                elif adversary_mine == 0 and match_win == 0:
                    N_a = 0
                    R_a = 0
                    N_h = 0
                    R_h = 0

                    memPools_h, block_reward = self.update_memPool_new_block(memPools_h, memPool_canonical, h)

                    rewards_h[h] = block_reward
                    h = h + 1
                    match = 0
                    latest = 1

        # abort
        elif 3 <= action <= 3 + self.len_abort:
            i = int(action - 3)
            if adversary_mine == 1:
                N_a = 0
                R_a = 0
                N_h = h - i
                R_h = self.total_reward(rewards_h[:h-i])
                d = h - i + a
                memPool_canonical = memPools_h[h-i-1].copy()
                memPools_a = np.zeros((self.max_fork, self.N_memPool_section))
                memPools_a, block_reward = self.update_memPool_new_block(memPools_a, memPool_canonical, 0)
                memPools_h = np.roll(memPools_h, -(h-i), axis=0)
                memPools_h[i:, ] = 0

                rewards_a = np.zeros(self.max_fork)
                rewards_h = np.roll(rewards_h, -(h - i))
                rewards_h[i:] = 0
                rewards_a[0] = block_reward
                a = 1
                h = i
                latest = 0
            else:
                N_a = 0
                R_a = 0
                N_h = h - i
                R_h = self.total_reward(rewards_h[:h - i])
                d =  h - i + a
                memPool_canonical = memPools_h[h - i - 1].copy()
                memPools_a = np.zeros((self.max_fork, self.N_memPool_section))
                memPools_h = np.roll(memPools_h, -(h - i), axis=0)
                memPools_h, block_reward = self.update_memPool_new_block(memPools_h, memPool_canonical, i)
                memPools_h[i+1:, ] = 0

                rewards_a = np.zeros(self.max_fork)
                rewards_h = np.roll(rewards_h, -(h - i))
                rewards_h[i] = block_reward
                rewards_h[i+1:] = 0
                a = 0
                h = i + 1
                latest = 1

        # block_undercutting
        elif action == 4 + self.len_abort:
            if adversary_mine == 1:
                N_a = 0
                R_a = 0
                N_h = h - 1
                R_h = self.total_reward(rewards_h[:h - 1])
                d = h - 1 + a
                if h > 1:
                    memPool_canonical = memPools_h[h - 2].copy()
                memPools_h = np.roll(memPools_h, -(h - 1), axis=0)
                memPools_h[1:, ] = 0

                rewards_a = np.zeros(self.max_fork)
                rewards_h = np.roll(rewards_h, - (h-1))
                rewards_h[1:] = 0

                memPools_a = np.zeros((self.max_fork, self.N_memPool_section))
                memPools_a, block_reward = self.update_memPool_undercut(memPools_a, memPools_h, memPool_canonical, 1, rewards_h, self.epsilon)
                rewards_a[0] = block_reward
                a = 1
                h = 1
                latest = 0
                undercut = 1
            else:
                N_a = 0
                R_a = 0
                N_h = h - 1
                R_h = self.total_reward(rewards_h[:h - 1])
                d = h - 1 + a
                if h > 1:
                    memPool_canonical = memPools_h[h - 2].copy()
                memPools_a = np.zeros((self.max_fork, self.N_memPool_section))
                memPools_h = np.roll(memPools_h, -(h - 1), axis=0)
                memPools_h, block_reward = self.update_memPool_new_block(memPools_h, memPool_canonical, 1)
                memPools_h[2:, ] = 0

                rewards_a = np.zeros(self.max_fork)
                rewards_h = np.roll(rewards_h, -(h - 1))
                rewards_h[1] = block_reward
                rewards_h[2:] = 0
                a = 0
                h = 2
                latest = 1

        # fork_undercutting
        elif action == 5 + self.len_abort:
            if adversary_mine == 1:
                N_a = 0
                R_a = 0
                N_h = 0
                R_h = 0

                memPools_a, block_reward = self.update_memPool_undercut(memPools_a, memPools_h, memPool_canonical, h, rewards_h, self.epsilon)

                rewards_a[h-1] = block_reward
                a = a + 1
                latest = 0
                undercut = 1
            else:
                N_a = 0
                R_a = 0
                N_h = 0
                R_h = 0

                memPools_h, block_reward = self.update_memPool_new_block(memPools_h, memPool_canonical, h)

                rewards_h[h] = block_reward
                h = h+1
                latest = 1
        # print('information', state_[0], state_[1], action,'memPool_canonical_after_block', memPool_canonical)
        state_ = np.concatenate(([a, h, match, latest, undercut], rewards_a, rewards_h, memPool_canonical), 0)
        for i in range(self.max_fork):
            state_ = np.concatenate((state_, memPools_a[i]), 0)
        for i in range(self.max_fork):
            state_ = np.concatenate((state_, memPools_h[i]), 0)
        return state_, N_a, R_a, N_h, R_h, d
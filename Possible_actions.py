import numpy as np

def selfish_mining_undercutting(state, env):
    state = np.array(state)
    a = int(state[0])
    h = int(state[1])
    match = state[2]
    latest = state[3]
    undercut = state[4]
    rewards_a = state[5: 5 + env.max_fork]
    rewards_h = state[5 + env.max_fork: 5 + 2 * env.max_fork]
    memPool_canonical = state[5 + 2 * env.max_fork: 5 + 2 * env.max_fork + env.N_memPool_section]
    state = state[5 + 2 * env.max_fork + env.N_memPool_section:]
    memPools_a = np.zeros((env.max_fork, env.N_memPool_section))
    for i in range(env.max_fork):
        memPools_a[i] = state[i * env.N_memPool_section: (i + 1) * env.N_memPool_section]

    state = state[env.max_fork * env.N_memPool_section:]
    memPools_h = np.zeros((env.max_fork, env.N_memPool_section))
    for i in range(env.max_fork):
        memPools_h[i] = state[i * env.N_memPool_section: (i + 1) * env.N_memPool_section]

    n_actions = 6 + env.len_abort
    possibleActionSet = np.zeros(n_actions)
    possibleActionSet[0:] = -1e20
    if h == env.max_fork:
        possibleActionSet[3] = 1
        return possibleActionSet
    if a == env.max_fork:
        possibleActionSet[0] = 1
        return possibleActionSet

    # undercut_possibility
    if undercut == 1:
        possibleActionSet[1] = 1
        return possibleActionSet

    if a > h:
        possibleActionSet[0] = 1
    if (env.connectivity > 0 or env.rational > 0) and (latest == 1) and (a >= h) and (h > 0):
        possibleActionSet[1] = 1
    possibleActionSet[2] = 1
    if h > a and latest == 1:
        possibleActionSet[3] = 1
    for i in range(1, env.len_abort + 1):
        if h > a + i and latest == 1:
            possibleActionSet[3+i] = 1
    if (env.rational > 0) and (h > 0) and (h > a) and (latest == 1):
        possibleActionSet[4 + env.len_abort] = 1
    if (env.rational > 0) and (h > 1) and (h == a+1) and (latest == 1):
        _, block_reward_h_fork = env.update_memPool_new_block(memPools_h, memPool_canonical, h)
        _, block_reward_a_fork = env.update_memPool_new_block(memPools_a, memPool_canonical, a)
        if block_reward_a_fork >= block_reward_h_fork + env.epsilon:
            possibleActionSet[5 + env.len_abort] = 1
    return possibleActionSet

def selfish_mining(state, env):
    state = np.array(state)
    a = int(state[0])
    h = int(state[1])
    match = state[2]
    latest = state[3]
    undercut = state[4]
    rewards_a = state[5: 5 + env.max_fork]
    rewards_h = state[5 + env.max_fork: 5 + 2 * env.max_fork]
    memPool_canonical = state[5 + 2 * env.max_fork: 5 + 2 * env.max_fork + env.N_memPool_section]
    state = state[5 + 2 * env.max_fork + env.N_memPool_section:]
    memPools_a = np.zeros((env.max_fork, env.N_memPool_section))
    for i in range(env.max_fork):
        memPools_a[i] = state[i * env.N_memPool_section: (i + 1) * env.N_memPool_section]

    state = state[env.max_fork * env.N_memPool_section:]
    memPools_h = np.zeros((env.max_fork, env.N_memPool_section))
    for i in range(env.max_fork):
        memPools_h[i] = state[i * env.N_memPool_section: (i + 1) * env.N_memPool_section]

    n_actions = 6 + env.len_abort
    possibleActionSet = np.zeros(n_actions)
    possibleActionSet[0:] = -1e20
    if h == env.max_fork:
        possibleActionSet[3] = 1
        return possibleActionSet
    if a == env.max_fork:
        possibleActionSet[0] = 1
        return possibleActionSet

    # undercut_possibility
    if undercut == 1:
        possibleActionSet[1] = 1
        return possibleActionSet

    if a > h:
        possibleActionSet[0] = 1
    if (env.connectivity > 0 or env.rational > 0) and (latest == 1) and (a >= h) and (h > 0):
        possibleActionSet[1] = 1
    possibleActionSet[2] = 1
    if h > a and latest == 1:
        possibleActionSet[3] = 1
    for i in range(1, env.len_abort + 1):
        if h > a + i and latest == 1:
            possibleActionSet[3+i] = 1
    # if (env.rational > 0) and (h > 0) and (h > a) and (latest == 1):
    #     possibleActionSet[4 + env.len_abort] = 1
    # if (env.rational > 0) and (h > 1) and (h == a+1) and (latest == 1):
    #     _, block_reward_h_fork = env.update_memPool_new_block(memPools_h, memPool_canonical, h)
    #     _, block_reward_a_fork = env.update_memPool_new_block(memPools_a, memPool_canonical, a)
    #     if block_reward_a_fork >= block_reward_h_fork + env.epsilon:
    #         possibleActionSet[5 + env.len_abort] = 1
    return possibleActionSet

def undercutting(state, env):
    state = np.array(state)
    a = int(state[0])
    h = int(state[1])
    match = state[2]
    latest = state[3]
    undercut = state[4]
    rewards_a = state[5: 5 + env.max_fork]
    rewards_h = state[5 + env.max_fork: 5 + 2 * env.max_fork]
    memPool_canonical = state[5 + 2 * env.max_fork: 5 + 2 * env.max_fork + env.N_memPool_section]
    state = state[5 + 2 * env.max_fork + env.N_memPool_section:]
    memPools_a = np.zeros((env.max_fork, env.N_memPool_section))
    for i in range(env.max_fork):
        memPools_a[i] = state[i * env.N_memPool_section: (i + 1) * env.N_memPool_section]

    state = state[env.max_fork * env.N_memPool_section:]
    memPools_h = np.zeros((env.max_fork, env.N_memPool_section))
    for i in range(env.max_fork):
        memPools_h[i] = state[i * env.N_memPool_section: (i + 1) * env.N_memPool_section]

    n_actions = 6 + env.len_abort
    possibleActionSet = np.zeros(n_actions)
    possibleActionSet[0:] = -1e20
    if h == env.max_fork:
        possibleActionSet[3] = 1
        return possibleActionSet
    if a == env.max_fork:
        possibleActionSet[0] = 1
        return possibleActionSet

    # undercut_possibility
    if undercut == 1:
        possibleActionSet[1] = 1
        return possibleActionSet

    if a > h:
        possibleActionSet[0] = 1
        return possibleActionSet
    # if (env.connectivity > 0 or env.rational > 0) and (latest == 1) and (a >= h) and (h > 0):
    #     possibleActionSet[1] = 1
    # possibleActionSet[2] = 1
    # if h > a and latest == 1:
    #     possibleActionSet[3] = 1
    # for i in range(1, env.len_abort + 1):
    #     if h > a + i and latest == 1:
    #         possibleActionSet[3+i] = 1
    if (env.rational > 0) and (h > 0) and (h > a) and (latest == 1):
        possibleActionSet[4 + env.len_abort] = 1
    # if (env.rational > 0) and (h > 1) and (h == a+1) and (latest == 1):
    #     _, block_reward_h_fork = env.update_memPool_new_block(memPools_h, memPool_canonical, h)
    #     _, block_reward_a_fork = env.update_memPool_new_block(memPools_a, memPool_canonical, a)
    #     if block_reward_a_fork >= block_reward_h_fork + env.epsilon:
    #         possibleActionSet[5 + env.len_abort] = 1
    return possibleActionSet
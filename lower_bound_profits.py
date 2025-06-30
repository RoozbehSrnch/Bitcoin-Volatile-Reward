import numpy as np
import mdptoolbox
import mdptoolbox.example
from scipy.sparse import lil_matrix


# Global variables
maxForkLen = None
numOfStates = None
r_ten = 1
r_twenty = None
r_thirty = None
non_active, active = 0, 1
ten, twenty, thirty = 0, 1, 2
# Actions: 1 adopt, 2 override, 3 match, 4 wait
choices = 4
adopt, override, match, wait = 0, 1, 2, 3

def st2stnum(a, a_array, h, t_last, match):
    if not ((0 <= a <= maxForkLen and isinstance(a, int)) and (len(a_array) == a and all(x in [0, 1, 2] for x in a_array)) and \
            (0 <= h <= maxForkLen and isinstance(h, int)) and (t_last in [0, 1, 2]) and (match in [0, 1])):
        raise ValueError('The state is not defined.')
    a_index = 3**a - 1
    for i in range(a):
        a_index += a_array[i] * (3**i)
    return a_index * (maxForkLen + 1) * 3 * 2 + h * 3 * 2 + t_last * 2 + match

def stnum2st(num):
    match = num % 2
    temp = num // 2
    t_last = temp % 3
    temp = temp // 3
    h = temp % (maxForkLen + 1)
    a_index = temp // (maxForkLen + 1)
    a_index += 1
    a = int(np.log2(a_index)/np.log2(3))
    a_index -= 3**a
    a_array = np.zeros(a, int)
    for i in range(a):
        a_array[i] = a_index % 3
        a_index = a_index // 3
    return a, a_array, h, t_last, match

def Reward(a_array):
    return np.sum(a_array == 2) * r_thirty/r_ten + np.sum(a_array == 1) * r_twenty/r_ten + np.sum(a_array == 0) * r_ten/r_ten


def lower_bound_selfish_mining_profitability(BTC_args, Mempool_args):
    global maxForkLen, numOfStates, r_twenty, r_thirty
    print('Obtaining an MDP-based lower bound for selfish mining profitability before difficulty adjustment ...', flush=True)
    alphaPower = BTC_args.adversarial_ratio
    gammaRatio = BTC_args.connectivity
    rationality = BTC_args.rational_ratio if BTC_args.epsilon == 0 else 0
    maxForkLen = min(BTC_args.max_fork, 7)
    r_twenty = r_ten * Mempool_args.fee_twenty_ten_ratio
    r_thirty = r_ten * Mempool_args.fee_thirty_ten_ratio
    numOfStates = (3**(maxForkLen + 1)-1) * (maxForkLen + 1) * 3 * 2
    print(f"numOfStates: {numOfStates}")
    P = [lil_matrix((numOfStates, numOfStates), dtype=np.float64) for _ in range(choices)]
    Rs = [lil_matrix((numOfStates, numOfStates), dtype=np.float64) for _ in range(choices)]
    Diff = [lil_matrix((numOfStates, numOfStates), dtype=np.float64) for _ in range(choices)]
    
    for i in range(numOfStates):
        if i % 100000 == 0:
            print(f"processing state: {i}")

        a, a_array, h, t_last, Match = stnum2st(i)
        P[adopt][i, st2stnum(1, [0], 0, ten, non_active)] = alphaPower
        Diff[adopt][i, st2stnum(1, [0], 0, ten, non_active)] = h
        P[adopt][i, st2stnum(0, [], 1, twenty, non_active)] = 1 - alphaPower
        Diff[adopt][i, st2stnum(0, [], 1, twenty, non_active)] = h

        # Define override
        if a > h:
            P[override][i, st2stnum(a - h, np.append(a_array[h+1:], t_last), 0, ten, non_active)] = alphaPower
            Rs[override][i, st2stnum(a - h, np.append(a_array[h+1:], t_last), 0, ten, non_active)] = Reward(a_array[:h+1])
            Diff[override][i, st2stnum(a - h, np.append(a_array[h+1:], t_last), 0, ten, non_active)] = h + 1
            P[override][i, st2stnum(a - h - 1, a_array[h+1:], 1, min(t_last + 1, thirty), non_active)] = 1 - alphaPower
            Rs[override][i, st2stnum(a - h - 1, a_array[h+1:], 1, min(t_last + 1, thirty), non_active)] = Reward(a_array[:h+1])
            Diff[override][i, st2stnum(a - h - 1, a_array[h+1:], 1, min(t_last + 1, thirty), non_active)] = h + 1
        else:  # Just for completeness
            P[override][i, 0] = 1
            Rs[override][i, 0] = - 10000
            Diff[override][i, 0] = 10000

        # Define wait
        if Match != active and a + 1 <= maxForkLen and h + 1 <= maxForkLen:
            # print(a, a_array, h, t_last, match)
            # print(a_array, [t_last], np.append(a_array, t_last))
            P[wait][i, st2stnum(a + 1, np.append(a_array, t_last), h, ten, non_active)] = alphaPower
            P[wait][i, st2stnum(a, a_array, h + 1, min(t_last + 1, thirty), non_active)] = 1 - alphaPower
        elif Match == active and a > h > 0 and a + 1 <= maxForkLen and h + 1 <= maxForkLen:
            P[wait][i, st2stnum(a + 1, np.append(a_array, t_last), h, ten, active)] = alphaPower
            P[wait][i, st2stnum(a - h, a_array[h:], 1, min(t_last + 1, thirty), non_active)] = rationality * (1 - alphaPower) + gammaRatio * (1 - rationality) * (1 - alphaPower)
            Rs[wait][i, st2stnum(a - h, a_array[h:], 1, min(t_last + 1, thirty), non_active)] = Reward(a_array[:h])
            Diff[wait][i, st2stnum(a - h, a_array[h:], 1, min(t_last + 1, thirty), non_active)] = h
            P[wait][i, st2stnum(a, a_array, h + 1, min(t_last + 1, thirty), non_active)] = (1 - gammaRatio) * (1 - rationality) * (1 - alphaPower)
        else:
            P[wait][i, 0] = 1
            Rs[wait][i, 0] = - 10000
            Diff[wait][i, 0] = 10000

        # Define match
        if t_last > ten and a >= h > 0 and a + 1 <= maxForkLen and h + 1 <= maxForkLen:
            P[match][i, st2stnum(a + 1, np.append(a_array, t_last), h, ten, active)] = alphaPower
            P[match][i, st2stnum(a - h, a_array[h:], 1, min(t_last + 1, thirty), non_active)] = rationality * (1 - alphaPower) + gammaRatio * (1 - rationality) * (1 - alphaPower)
            Rs[match][i, st2stnum(a - h, a_array[h:], 1, min(t_last + 1, thirty), non_active)] = Reward(a_array[:h])
            Diff[match][i, st2stnum(a - h, a_array[h:], 1, min(t_last + 1, thirty), non_active)] = h
            P[match][i, st2stnum(a, a_array, h + 1, min(t_last + 1, thirty), non_active)] = (1 - gammaRatio) * (1 - rationality) * (1 - alphaPower)
        else:
            P[match][i, 0] = 1
            Rs[match][i, 0] = - 10000
            Diff[match][i, 0] = 10000


    P = [matrix.tocsr(copy=True) for matrix in P]
    Rs = [matrix.tocsr(copy=True) for matrix in Rs]
    Diff = [matrix.tocsr(copy=True) for matrix in Diff]

    epsilon = 0.000001
    mdp = mdptoolbox.mdp.RelativeValueIteration(P, Rs, epsilon / 8)
    mdp.run()
    reward_before_difficulty_adjustment = mdp.average_reward

    return reward_before_difficulty_adjustment

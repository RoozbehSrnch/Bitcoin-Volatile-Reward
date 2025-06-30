# import numpy as np
# import mdptoolbox
# import mdptoolbox.example
# from scipy.sparse import lil_matrix
import numpy as np
import mdptoolbox
from mdptoolbox.util import check
import mdptoolbox.example
from scipy.sparse import spmatrix, csr_matrix, lil_matrix, identity
import inspect
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import math
from numbers import Integral

# Global variables
maxForkLen = None
numOfStates = None
maxTotalTimeUnit = 61
maxTimeUnit = 61
irrelevant, relevant, active = 0, 1, 2
numTimeSection = 21
mining_rate = 0.1
frequency_deviation = 0.001
# Actions: 1 adopt, 2 override, 3 match, 4 wait
choices = 4
adopt, override, match, wait = 0, 1, 2, 3
action_names = {
    adopt: "adopt",
    override: "override",
    match: "match",
    wait: "wait"
}

def array_mining_pr_time(frequency_deviation):
    """ Find the time t such that the probability of no event occurring before t is equal to the given probability p."""

    if not (0 < frequency_deviation <= 1):
        raise ValueError("Probability p must be in the range (0, 1].")
    t = -math.log(frequency_deviation) / mining_rate
    """ Dividing time into discrete sections. p_success is the probability of a successful mining in each section."""
    timeSection = t / (numTimeSection-1)
    p_success = 1 - math.exp(- mining_rate * timeSection)
    pr = np.zeros(numTimeSection)
    for i in range(numTimeSection - 1):
        pr[i] = p_success * (1-p_success) ** i
    pr[-1] = 1 - np.sum(pr)
    time_passage = np.arange(0, numTimeSection)
    t_average_section, time_passage_average = time_passage_array(timeSection, mining_rate, numTimeSection)
    average_block_generation_time2 = np.dot(pr, time_passage) * timeSection
    return pr, time_passage, average_block_generation_time2, timeSection

def time_passage_array(timeSection, rate, numTimeSection):
    lambda_ = rate
    denom = 1 - math.exp(-lambda_ * timeSection)
    term1 = - timeSection * math.exp(-lambda_ * timeSection)
    term2 = (1 / lambda_) * (1 - math.exp(-lambda_ * timeSection))
    numer = term1 + term2
    t_average_section = numer / denom
    time_passage_average = np.zeros(numTimeSection)
    for i in range(numTimeSection):
        time_passage_average[i] = t_average_section + i * timeSection
    return t_average_section, time_passage_average

def st2stnum(a, h, delta_total, delta_last, fork):
    if not ((0 <= a <= maxForkLen and isinstance(a, Integral)) and (0 <= h <= maxForkLen and isinstance(h, Integral)) and
            (0 <= delta_total <= maxTotalTimeUnit and isinstance(delta_total, Integral)) and (0 <= delta_last <= maxTimeUnit and isinstance(delta_last, Integral)) and (fork in [0, 1, 2])):
        print(type(delta_total), type(delta_last))
        raise ValueError(f'The state {a, h, delta_total, delta_last, fork} is invalid.')
    state_number = 3 * (maxTimeUnit + 1) * (maxTotalTimeUnit + 1) * (maxForkLen + 1) * a + 3 * (maxTimeUnit + 1) * (maxTotalTimeUnit + 1) * h + 3 * (maxTimeUnit + 1) * delta_total + 3 * delta_last + fork
    return state_number

def stnum2st(num):
    if not (0 <= num < numOfStates and isinstance(num, int)):
        raise ValueError(f'The state number {num} is invalid.')
    fork = num % 3
    temp = num // 3
    delta_last = temp % (maxTimeUnit + 1)
    temp = temp // (maxTimeUnit + 1)
    delta_total = temp % (maxTotalTimeUnit + 1)
    temp = temp // (maxTotalTimeUnit + 1)
    h = temp % (maxForkLen + 1)
    temp = temp // (maxForkLen + 1)
    a = temp % (maxForkLen + 1)
    return a, h, delta_total, delta_last, fork

def print_state(policy, a, h, delta_total, delta_last, fork):
    print(f'State: a= {a}, h= {h}, delta_total= {delta_total}, delta_last= {delta_last}, fork= {fork}; '
          f'Action: {action_names[policy[st2stnum(a, h, delta_total, delta_last, fork)]]}')
    
def threshold(gamma, intercept, slope, frequency_deviation):
    higher_alpha = 0.5
    lower_alpha = 0
    epsilon = 1e-3
    while higher_alpha - lower_alpha > epsilon:
        alpha = (higher_alpha + lower_alpha)/2
        reward_ratio, policy, honest_reward = selfish_mining(alpha, gamma, intercept, slope, frequency_deviation)
        condition1 = reward_ratio - honest_reward > 1e-6
        condition2 = any(policy[st2stnum(0, 1, 0, delta, 1)] == wait for delta in range(numTimeSection))
        condition3 = any(policy[st2stnum(1, 0, delta, 0, 0)] == wait for delta in range(numTimeSection))
        print(reward_ratio, honest_reward, condition1, condition2, condition3)
        if condition1 and (condition2 or condition3):
            higher_alpha = alpha
            print(f"For the adversary with mining power {alpha}, selfish mining is PROFITABLE.")
            print('###################################')
        else:
            lower_alpha = alpha
            print(f"For the adversary with mining power {alpha}, selfish mining is NOT profitable.")
            print('###################################')

    return higher_alpha


def lower_bound_selfish_mining_profitability2(BTC_args, Mempool_args):
    global maxForkLen, numOfStates, r_twenty, r_thirty
    print('Obtaining an MDP-based lower bound for selfish mining profitability before difficulty adjustment ...', flush=True)
    alphaPower = BTC_args.adversarial_ratio
    gammaRatio = BTC_args.connectivity
    rationality = BTC_args.rational_ratio if BTC_args.epsilon == 0 else 0
    maxForkLen = min(BTC_args.max_fork, 6)
    slope = Mempool_args.slope
    intercept = Mempool_args.intercept
    numOfStates = (maxForkLen + 1) ** 2 * (maxTotalTimeUnit + 1) * (maxTimeUnit + 1) * 3
    print(f"numOfStates: {numOfStates}")
    P = [lil_matrix((numOfStates, numOfStates), dtype=np.float64) for _ in range(choices)]
    N_a = [lil_matrix((numOfStates, numOfStates), dtype=np.float64) for _ in range(choices)]
    T_a = [lil_matrix((numOfStates, numOfStates), dtype=np.float64) for _ in range(choices)]
    Diff = [lil_matrix((numOfStates, numOfStates), dtype=np.float64) for _ in range(choices)]

    Pr, time_passage, average_block_generation_time, timeSection = array_mining_pr_time(frequency_deviation)
    for i in range(numOfStates):
        if i % 10000 == 0:
            print(f"processing state: {i}")
        a, h, delta_total, delta_last, fork = stnum2st(i)

        for t in range(numTimeSection):
            P[adopt][i, st2stnum(1, 0, time_passage[t], 0, irrelevant)] = alphaPower * Pr[t]
            Diff[adopt][i, st2stnum(1, 0, time_passage[t], 0, irrelevant)] = h

            P[adopt][i, st2stnum(0, 1, 0, time_passage[t], relevant)] = (1 - alphaPower) * Pr[t]
            Diff[adopt][i, st2stnum(0, 1, 0, time_passage[t], relevant)] = h


        # Override
        if a == h + 1 and delta_last + delta_total <= maxTotalTimeUnit and delta_last <= maxTimeUnit:
            max_delta = min(numTimeSection, maxTotalTimeUnit + 1 - delta_last)
            pr = Pr[:max_delta-1]
            pr = np.append(pr, sum(Pr[max_delta-1:]))
            # print(np.sum(pr), pr)
            for t in range(max_delta):
                P[override][i, st2stnum(1, 0, delta_last + time_passage[t], 0, irrelevant)] = alphaPower * pr[t]
                N_a[override][i, st2stnum(1, 0, delta_last + time_passage[t], 0, irrelevant)] = a
                T_a[override][i, st2stnum(1, 0, delta_last + time_passage[t], 0, irrelevant)] = delta_total
                Diff[override][i, st2stnum(1, 0, delta_last + time_passage[t], 0, irrelevant)] = a

            max_delta = min(numTimeSection, maxTimeUnit + 1 - delta_last)
            pr = Pr[:max_delta - 1]
            pr = np.append(pr,sum(Pr[max_delta - 1:]))
            for t in range(max_delta):
                P[override][i, st2stnum(0, 1, 0, delta_last + time_passage[t], relevant)] = (1 - alphaPower) * pr[t]
                N_a[override][i, st2stnum(0, 1, 0, delta_last + time_passage[t], relevant)] = a
                T_a[override][i, st2stnum(0, 1, 0, delta_last + time_passage[t], relevant)] = delta_total
                Diff[override][i, st2stnum(0, 1, 0, delta_last + time_passage[t], relevant)] = a
        elif a > h + 1 and delta_last + delta_total <= maxTotalTimeUnit and delta_last <= maxTimeUnit:
            max_delta = min(numTimeSection, maxTotalTimeUnit + 1 - delta_last - delta_total)
            pr = Pr[:max_delta - 1]
            pr = np.append(pr,sum(Pr[max_delta - 1:]))
            for t in range(max_delta):
                P[override][i, st2stnum(a-h, 0, delta_total + delta_last + time_passage[t], 0, irrelevant)] = alphaPower * pr[t]
                N_a[override][i, st2stnum(a-h, 0, delta_total + delta_last + time_passage[t], 0, irrelevant)] = h + 1
                Diff[override][i, st2stnum(a-h, 0, delta_total + delta_last + time_passage[t], 0, irrelevant)] = h + 1

            max_delta = min(numTimeSection, maxTimeUnit + 1 - delta_last)
            pr = Pr[:max_delta - 1]
            pr = np.append(pr, sum(Pr[max_delta - 1:]))
            for t in range(max_delta):
                P[override][i, st2stnum(a - h - 1, 1, delta_total, delta_last + time_passage[t], relevant)] = (1 - alphaPower) * pr[t]
                N_a[override][i, st2stnum(a - h - 1, 1, delta_total, delta_last + time_passage[t], relevant)] = h + 1
                Diff[override][i, st2stnum(a - h - 1, 1, delta_total, delta_last + time_passage[t], relevant)] = h + 1

        else:  # Just for completeness
            P[override][i, 0] = 1
            Diff[override][i, 0] = 10000
            N_a[override][i, 0] = -10000
            T_a[override][i, 0] = -10000

        # Wait
        if fork != active and a + 1 <= maxForkLen and h + 1 <= maxForkLen and delta_last + delta_total <= maxTotalTimeUnit and delta_last <= maxTimeUnit:
            max_delta = min(numTimeSection, maxTotalTimeUnit + 1 - delta_last - delta_total)
            pr = Pr[:max_delta - 1]
            pr = np.append(pr, sum(Pr[max_delta - 1:]))
            for t in range(max_delta):
                P[wait][i, st2stnum(a + 1, h, delta_total + delta_last + time_passage[t], 0, irrelevant)] = alphaPower * pr[t]

            max_delta = min(numTimeSection, maxTimeUnit + 1 - delta_last)
            pr = Pr[:max_delta - 1]
            pr = np.append(pr,sum(Pr[max_delta - 1:]))
            for t in range(max_delta):
                P[wait][i, st2stnum(a, h + 1, delta_total, delta_last + time_passage[t], relevant)] = (1 - alphaPower) * pr[t]
        elif fork == active and a > h and h > 0 and a + 1 <= maxForkLen and h + 1 <= maxForkLen and delta_last + delta_total <= maxTotalTimeUnit and delta_last <= maxTimeUnit:
            max_delta = min(numTimeSection, maxTotalTimeUnit + 1 - delta_last - delta_total)
            pr = Pr[:max_delta - 1]
            pr = np.append(pr,sum(Pr[max_delta - 1:]))
            for t in range(max_delta):
                P[wait][i, st2stnum(a + 1, h, delta_total + delta_last + time_passage[t], 0, active)] = alphaPower * pr[t]

            max_delta = min(numTimeSection, maxTimeUnit + 1 - delta_last)
            pr = Pr[:max_delta - 1]
            pr = np.append(pr, sum(Pr[max_delta - 1:]))
            for t in range(max_delta):
                P[wait][i, st2stnum(a - h, 1, delta_total, delta_last + time_passage[t], relevant)] = gammaRatio * (1 - alphaPower) * pr[t]
                N_a[wait][i, st2stnum(a - h, 1, delta_total, delta_last + time_passage[t], relevant)] = h
                Diff[wait][i, st2stnum(a - h, 1, delta_total, delta_last + time_passage[t], relevant)] = h

                P[wait][i, st2stnum(a, h + 1, delta_total, delta_last + time_passage[t], relevant)] = (1 - gammaRatio) * (1 - alphaPower) * pr[t]
        else:
            P[wait][i, 0] = 1
            Diff[wait][i, 0] = 10000
            N_a[wait][i, 0] = -10000
            T_a[wait][i, 0] = -10000

        # Match
        if fork == relevant and a == h and h > 0 and a + 1 <= maxForkLen and h + 1 <= maxForkLen and delta_last + delta_total <= maxTotalTimeUnit and delta_last <= maxTimeUnit:
            max_delta = min(numTimeSection, maxTotalTimeUnit + 1 - delta_last - delta_total)
            # max_delta = max(max_delta, 1)
            pr = Pr[:max_delta - 1]
            pr = np.append(pr,sum(Pr[max_delta - 1:]))

            for t in range(max_delta):
                P[match][i, st2stnum(a + 1, h, delta_total + delta_last + time_passage[t], 0, active)] = alphaPower * pr[t]


            max_delta = min(numTimeSection, maxTimeUnit + 1 - delta_last)
            pr = Pr[:max_delta - 1]
            pr = np.append(pr,sum(Pr[max_delta - 1:]))
            for t in range(max_delta):
                P[match][i, st2stnum(0, 1, 0, delta_last + time_passage[t], relevant)] = gammaRatio * (1 - alphaPower) * pr[t]
                N_a[match][i, st2stnum(0, 1, 0, delta_last + time_passage[t], relevant)] = h
                T_a[match][i, st2stnum(0, 1, 0, delta_last + time_passage[t], relevant)] = delta_total
                Diff[match][i, st2stnum(0, 1, 0, delta_last + time_passage[t], relevant)] = h
                P[match][i, st2stnum(a, h + 1, delta_total, delta_last + time_passage[t], relevant)] = (1 - gammaRatio) * (1 - alphaPower) * pr[t]


        elif fork == relevant and a > h and h > 0 and a + 1 <= maxForkLen and h + 1 <= maxForkLen and delta_last + delta_total <= maxTotalTimeUnit and delta_last <= maxTimeUnit:
            max_delta = min(numTimeSection, maxTotalTimeUnit + 1 - delta_last - delta_total)
            pr = Pr[:max_delta - 1]
            pr = np.append(pr,sum(Pr[max_delta - 1:]))
            for t in range(max_delta):
                P[match][i, st2stnum(a + 1, h, delta_total + delta_last + time_passage[t], 0, active)] = alphaPower * pr[t]

            max_delta = min(numTimeSection, maxTimeUnit + 1 - delta_last)
            pr = Pr[:max_delta - 1]
            pr = np.append(pr, sum(Pr[max_delta - 1:]))
            for t in range(max_delta):
                P[match][i, st2stnum(a - h, 1, delta_total, delta_last + time_passage[t], relevant)] = gammaRatio * (
                            1 - alphaPower) * pr[t]
                N_a[match][i, st2stnum(a - h, 1, delta_total, delta_last + time_passage[t], relevant)] = h
                Diff[match][i, st2stnum(a - h, 1, delta_total, delta_last + time_passage[t], relevant)] = h

                P[match][i, st2stnum(a, h + 1, delta_total, delta_last + time_passage[t], relevant)] = (1 - gammaRatio) * (1 - alphaPower) * pr[t]


        else:
            P[match][i, 0] = 1
            Diff[match][i, 0] = 10000
            N_a[match][i, 0] = -10000
            T_a[match][i, 0] = -10000

    epsilon = 0.000001
    honest_reward = alphaPower * (intercept + slope * average_block_generation_time)
    Wrou = [intercept * N_a[i] + slope * (T_a[i] * timeSection) for i in range(choices)]
    mdp = mdptoolbox.mdp.RelativeValueIteration(P, Wrou, epsilon)
    mdp.run()
    reward = mdp.average_reward
    return reward/honest_reward

    

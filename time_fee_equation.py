from environment import Bitcoin_Transaction_Fee_Model
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

def choose_action_honest(state):
    if state[0] == 1:
        action = 0
    else:
        if state[0] != 0 or state[1] != 1:
            raise Exception("This is not possible.")
        action = 3
    return action

def curve_func(x, a, b, c, d):
    return a * np.log(x**b + c) + d


def time_fee(BTC_args, Mempool_args):
    print('Obtaining smaple points for time-fee equation ...', flush=True)
    mining_time = BTC_args.mining_time
    adversarial_ratio = BTC_args.adversarial_ratio
    BTC_args.mining_time = 10
    BTC_args.adversarial_ratio = 1
    env = Bitcoin_Transaction_Fee_Model(BTC_args, Mempool_args)
    state = env.reset()
    block_rewards = []
    block_generation_times = []
    cnt = 0
    while cnt <= Mempool_args.N_steps_honest_mining:
        action = choose_action_honest(state)
        state_, n_a, n_h, adversary_reward, duration,_ = env.step(action, 0)
        block_rewards.append(adversary_reward)
        block_generation_times.append(duration)
        state = state_
        cnt += 1
        if cnt % 50000 == 0:
            print("Number of steps:", cnt, flush=True)
    block_rewards = block_rewards[1:]
    block_generation_times = block_generation_times[:-1]
    
    # Create the scatter plot
    plt.scatter(block_generation_times, block_rewards, color='blue', label='Bitcoin block data', zorder=3)
    
    # The linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(block_generation_times, block_rewards)
    a_1 = slope
    a_2 = intercept
    y_30 = a_1 * 30 + a_2
    y_20 = a_1 * 20 + a_2
    y_10 = a_1 * 10 + a_2
    ratio_20_10 = y_20 / y_10
    ratio_30_10 = y_30 / y_10
    print("************************** Linear regression****************************")
    print(f"Time-fee equation is: f(t) = {a_1:.6f} * t + {a_2:.6f}", flush=True)
    x_vals = np.linspace(min(block_generation_times), max(block_generation_times), 100)
    y_vals = a_1 * x_vals + a_2
    plt.plot(x_vals, y_vals, color='red', label='Regression line', zorder=6)
    
    # The curve regression
    params, covariance = curve_fit(curve_func, block_generation_times, block_rewards)
    a, b, c, d = params
    y_30 = curve_func(30, a, b, c, d)
    y_20 = curve_func(20, a, b, c, d)
    y_10 = curve_func(10, a, b, c, d)
    ratio_20_10 = y_20 / y_10
    ratio_30_10 = y_30 / y_10
    print("************************** Curve regression****************************")
    print(f"Time-fee equation is: y(x) = {a:.6f} * ln(t^({b:.6f}) + {c:.6f}) + {d:.6f}", flush=True)
    x_fit = np.linspace(min(block_generation_times), max(block_generation_times), 100)
    y_fit = curve_func(x_fit, a, b, c, d)
    plt.plot(x_fit, y_fit, color='orange', label='Regression curve', zorder=6)

    # Add labels and title
    plt.xlabel('Block generation time (minutes)', fontsize=13)
    plt.ylabel('Total transaction fees per block (BTCs)', fontsize=13)
    plt.legend(fontsize=13)
    plt.grid(zorder=0)
    plt.xlim(0, 50)
    plt.show()
    BTC_args.mining_time = mining_time
    BTC_args.adversarial_ratio = adversarial_ratio

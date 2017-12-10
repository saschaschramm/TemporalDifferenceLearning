"""
n-step TD prediction
"""

import math
import random
from collections import defaultdict
import numpy as np
from Environment.catch import Catch

epsiodes = 1000
size = 4
env = Catch(size)
learning_rate = 0.1
state_values = defaultdict(lambda: 0)

def compute_state_values():
    random.seed(3)
    discount_rate = 0.9
    n = 2
    rewards = []
    states = []

    for episode in range(0, epsiodes):
        state = env.reset()
        T = math.inf
        t = 0
        states.append(state)

        while True:

            if t < T:
                # all actions equally likely
                action = random.randint(0, 3)
                next_state, reward, done = env.step(action)

                rewards.append(reward)
                states.append(next_state)

                if done:
                    T = t + 1

            state_index = t - n + 1

            if state_index >= 0: # check if we have enough states to compute total return
                total_return = 0
                for i in range(state_index, min(state_index+n, T)):
                    total_return += pow(discount_rate,i-state_index) * rewards[i]
                if state_index + n < T:
                    total_return += pow(discount_rate, n) * state_values[tuple(next_state.flatten())]

                update_state_value(total_return, states[state_index])

            if state_index == T - 1:
                # -> after terminal state we wait n-steps before we break.
                rewards = []
                states = []
                break
            t += 1

def update_state_value(total_return, state):
    td_error = total_return - state_values[tuple(state.flatten())]
    state_values[tuple(state.flatten())] = state_values[tuple(state.flatten())] + learning_rate * td_error

def render_state_values():
    grid = np.zeros(size*size)
    for key, value in state_values.items():
        grid = np.add(grid, np.multiply(key, value))

    for y in range(0, size):
        print()
        for x in range(0, size):
            index = y * size + x
            print('{0:.2f} '.format(grid[index]), end='')


"""
Example (n = 1):

t = 0:
S1, R1, not_done = step()
total_return = discount_rate^0 * R1 + discount_rate^1 * V(S1)
V(S0) = V(S0) + learning_rate * (total_return-V(S0))

t = 1:
S2, R2, done = step()
total_return = discount_rate^0 * R2
V(S1) = V(S1) + learning_rate * (total_return-V(S1))
"""

compute_state_values()
render_state_values()

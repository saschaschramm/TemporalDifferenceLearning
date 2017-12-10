"""
1-step TD prediction
"""

import random
from collections import defaultdict
import numpy as np
from Environment.catch import Catch

epsiodes = 1000
size = 4
env = Catch(size)
discount_rate = 0.9
learning_rate = 0.1
state_values = defaultdict(lambda: 0)

def compute_state_values():
    random.seed(3)
    for episode in range(0, epsiodes):
        state = env.reset()
        while True:
            # all actions equally likely
            action = random.randint(0, 3)
            next_state, reward, done = env.step(action)
            td_target = reward + discount_rate * state_values[tuple(next_state.flatten())]
            update_state_value(td_target, state)
            state = next_state

            if done:
                break

def update_state_value(td_target, state):
    td_error = td_target - state_values[tuple(state.flatten())]
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


compute_state_values()
render_state_values()
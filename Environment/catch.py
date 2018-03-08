import numpy as np
from Environment.catch_agent import CatchAgent

class Catch:

    def __init__(self, size):
        self.cols = size
        self.rows = size
        self.player = CatchAgent()
        self.enemy = CatchAgent()
        self.action_space = 4
        self.observation_space = (self.rows, self.cols, 1)

    def step(self, action):
        reward = 0.0
        done = False

        self.player.move(action)

        if self._check_wall(self.player.position):
            self.player.position = self.player.position_old

        if self._check_wall(self.enemy.position):
            self.enemy.position = self.enemy.position_old

        if self._check_hit():
            reward = 1.0
            done = True

        self._update()

        observation = self.pixels
        return observation, reward, done

    def _check_hit(self):
        return self.player.position == self.enemy.position

    def _check_wall(self, position):
        y = position[0]
        x = position[1]

        if (y >= self.rows) or (y < 0) or (x >= self.cols) or (x < 0):
            return True
        else:
            return False

    def _update(self):
        self.pixels = np.zeros(self.observation_space)
        self.pixels[self.player.position] = 1.0

    def reset(self):
        self.player.position = (0, 0)
        self.enemy.position = (self.rows-1, self.cols-1)
        self._update()
        return self.pixels

    def render(self):
        print(self.pixels.reshape((self.cols, self.rows)))
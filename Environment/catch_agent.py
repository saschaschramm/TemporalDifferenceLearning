class CatchAgent:

    def __init__(self):
        self.actions = {
            0: (0, -1),
            1: (0, 1),
            2: (-1, 0),
            3: (1, 0)
        }

        self.position = (0, 0)
        self.position_old = None

    def move(self, action):
        self.position_old = self.position
        d = self.delta(action)
        self.position = (self.position[0] + d[0], self.position[1] + d[1])

    def delta(self, action):
        return self.actions[action]
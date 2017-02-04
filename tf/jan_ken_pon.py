import os
import numpy as np


class JanKenPon:

    def __init__(self):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.player_hand = None
        self.opponent_hand = None
        self.observation = np.zeros(6)
        self.frame_length = 1
        self.state = np.stack([self.observation for _ in xrange(self.frame_length)], axis=0)
        self.enable_actions = (0, 1, 2, 3, 4, 5)
        self.enable_valid_actions = (0, 2, 5)

        # variables
        self.reset()

    def execute_action(self, action):
        """
        action:
            0: goo
            1: undefined
            2: choki
            3: undefined
            4: undefined
            5: par
        """
        # update player state
        self.player_hand = action

        # determine win or loose
        self.reward = 0
        self.terminal = False

        undefined = (1, 3, 4)

        if self.player_hand in undefined:
            self.reward = -3
        elif self.player_hand == self.opponent_hand:
            # self.reward = 0
            self.reward = -1
        elif self.player_hand == 0 and self.opponent_hand == 2:
            self.reward = 1
        elif self.player_hand == 2 and self.opponent_hand == 5:
            self.reward = 1
        elif self.player_hand == 5 and self.opponent_hand == 0:
            self.reward = 1
        else:
            self.reward = -1

        if self.reward != 0:
            self.terminal = True

    def observe(self):
        # set opponent next hand
        self.observation = np.zeros(6)
        self.opponent_hand = self.enable_valid_actions[np.random.randint(len(self.enable_valid_actions))]
        self.observation[self.opponent_hand] = 1

        # update state history
        self.update_state_history()

        return self.state, self.reward, self.terminal

    def update_state_history(self):
        self.state = np.append(self.state[1:], self.observation.reshape(-1, self.observation.size), axis=0)

    def render(self):
        pass

    def render_hand_shape(self, action):
        if action == 0:
            return 'goo'
        elif action == 2:
            return 'choki'
        elif action == 5:
            return 'par'
        return 'unknown'

    def reset(self):
        # reset player
        self.player_hand = np.random.randint(len(self.enable_actions))

        # reset opponent
        self.opponent_hand = self.enable_valid_actions[np.random.randint(len(self.enable_valid_actions))]

        # reset other variables
        self.reward = 0
        self.terminal = False

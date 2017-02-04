from collections import deque
import os

import numpy as np
import keras
from dqn_net import DQNNet


class DQNAgent():
    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, enable_actions, environment_name):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.environment_name = environment_name
        self.enable_actions = enable_actions
        self.n_actions = len(self.enable_actions)
        self.minibatch_size = 32
        self.replay_memory_size = 1000
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.exploration = 0.1
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.h5".format(self.environment_name)

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

        # model
        self.net = DQNNet(6, self.n_actions)
        self.init_model()

        # variables
        self.current_loss = 0.0

    def init_model(self):
        # Set up an optimizer
        optimizer = keras.optimizers.RMSprop(decay=self.learning_rate)
        self.net.model.summary()
        self.net.model.compile(loss='mean_squared_error',
                               optimizer=optimizer,
                               metrics=[])

    def Q_values(self, state):
        # Q(state, action) of all actions
        x = state.reshape(-1, state.size)
        pred = self.net.model.predict(x, batch_size=self.minibatch_size)
        return pred[0]

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            # random
            self.is_greedy_action = False
            return np.random.choice(self.enable_actions)
        else:
            # max_action Q(state, action)
            self.is_greedy_action = True
            return self.enable_actions[np.argmax(self.Q_values(state))]

    def store_experience(self, state, action, reward, state_1, terminal):
        self.D.append((state, action, reward, state_1, terminal))

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = self.enable_actions.index(action_j)

            y_j = self.Q_values(state_j)

            if terminal:
                y_j[action_j_index] = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                y_j[action_j_index] = reward_j + self.discount_factor * np.max(self.Q_values(state_j_1))  # NOQA

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)

        # Set up a trainer
        # Run the training
        state_size = state_minibatch[0].size
        x = np.array(state_minibatch).reshape(-1, state_size)
        y = np.array(y_minibatch).reshape(-1, state_size)
        self.net.model.fit(x, y, batch_size=self.minibatch_size, nb_epoch=1, verbose=1)

        # for log
        x = np.array(state_minibatch).reshape(-1, state_size)
        t = np.array(y_minibatch).reshape(-1, state_size)
        self.current_loss = self.net.model.evaluate(x, t, batch_size=self.minibatch_size)

    def load_model(self, model_path=None):
        if model_path:
            self.net.model = keras.models.load_model(model_path)
        else:
            self.net.model = keras.models.load_model('%s/%s' % (self.model_dir, self.model_name))
        self.init_model()

    def save_model(self):
        self.net.model.save('%s/%s' % (self.model_dir, self.model_name))

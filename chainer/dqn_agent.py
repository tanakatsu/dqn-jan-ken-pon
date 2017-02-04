from collections import deque
import os

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer.datasets import tuple_dataset
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
        self.model_name = "{}.npz".format(self.environment_name)

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

        # model
        self.model = L.Classifier(DQNNet(self.n_actions), lossfun=F.mean_squared_error)
        self.model.compute_accuracy = False

        # variables
        self.current_loss = 0.0

        # Set up an optimizer
        self.optimizer = chainer.optimizers.RMSprop()
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(self.learning_rate))

    def Q_values(self, state):
        # Q(state, action) of all actions
        x = state.reshape(-1, state.size)  # flatten
        return self.model.predictor(x).data[0]

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

        train = tuple_dataset.TupleDataset(state_minibatch, y_minibatch)
        train_iter = chainer.iterators.SerialIterator(train, minibatch_size)

        # Set up a trainer
        updater = training.StandardUpdater(train_iter, self.optimizer)
        trainer = training.Trainer(updater, (1, 'epoch'))

        # Take a snapshot
        # trainer.extend(extensions.snapshot_object(self.model, 'model_iter_{.updater.iteration}'), (1, 'epoch'))

        # for log
        # trainer.extend(extensions.LogReport())
        # trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time']))
        # trainer.extend(extensions.ProgressBar())

        # Run the training
        trainer.run()

        # for log
        state_size = state_minibatch[0].size
        x = np.array(state_minibatch).reshape(-1, state_size)  # flatten
        t = np.array(y_minibatch).reshape(-1, state_size)  # flatten
        self.current_loss = self.model(x, t).data

    def load_model(self, model_path=None):
        if model_path:
            serializers.load_npz(model_path, self.model)
        else:
            serializers.load_npz('%s/%s' % (self.model_dir, self.model_name), self.model)

    def save_model(self):
        serializers.save_npz('%s/%s' % (self.model_dir, self.model_name), self.model)

import chainer
import chainer.functions as F
import chainer.links as L


class DQNNet(chainer.Chain):

    """
    Multi Layer Perceptron with Experience Replay
    """

    def __init__(self, n_out):
        super(DQNNet, self).__init__(
            l1=L.Linear(None, 16),     # n_in -> n_units
            l2=L.Linear(None, n_out)  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l2(h1)

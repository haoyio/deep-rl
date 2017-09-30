import numpy as np


class Policy(object):
    """Abstract base class for action selection policies."""
    def select_action(self, **kwargs):
        raise NotImplementedError()


class EpsilonGreedyPolicy(Policy):
    """Exponentially-decaying epsilon-greedy policy."""
    def __init__(self, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99):
        super(EpsilonGreedyPolicy, self).__init__()
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def select_action(self, q_values):
        assert q_values.ndim == 1
        n_actions = q_values.shape[0]

        if np.random.uniform() < self.epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(q_values)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return action

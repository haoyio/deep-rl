import numpy as np

from deeprl.core import Agent
from deeprl.memory import Memory
from deeprl.policy import EpsilonGreedyPolicy


class DQNAgent(Agent):
    """Deep Q-learning based reinforcement learning agent."""

    def __init__(self,
                 model,
                 memory_limit=100000,
                 epsilon=1.0,
                 epsilon_min=0.1,
                 epsilon_decay=0.99,
                 discount=0.99,
                 batch_size=32,
                 replay_start_size=50000,
                 **kwargs):

        super(DQNAgent, self).__init__(**kwargs)

        self.model = model
        self.memory = Memory(memory_limit)
        self.policy = EpsilonGreedyPolicy(
            epsilon,
            epsilon_min,
            epsilon_decay
        )

        self.discount = discount  # discount factor
        self.batch_size = batch_size  # minibatch size
        self.replay_start_size = replay_start_size

    def act(self, observation, is_train=False):
        """Returns an action based on epsilon-greedy selection.

        If not |is_train|, we return the best action based on the model trained.
        """
        q_values = self.model.predict(observation)[0]

        if is_train:
            action = self.policy.select_action(q_values)
        else:
            action = np.argmax(q_values)

        return action

    def update(self):
        """Experience replay.

        Replays samples in the agent's memory to train the model and returns
        training history for model after replay.
        """
        if len(self.memory) < self.replay_start_size:
            return None

        observations = []
        targets = []

        minibatch = self.memory.sample(self.update_batch_size)

        for exp in minibatch:
            target = exp.reward
            if not exp.done:
                target = exp.reward + self.discount * \
                         np.amax(self.model.predict(exp.next_observation)[0])

            target_f = self.model.predict(exp.observation)[0]
            target_f[exp.action] = target

            observations.append(exp.observation[0])
            targets.append(target_f)

        history = self.model.fit(
            np.array(observations),
            np.array(targets),
            epochs=1,
            verbose=0
        )

        return history

    def remember(self, observation, action, reward, next_observation, done):
        self.memory.remember(observation, action, reward, next_observation, done)

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

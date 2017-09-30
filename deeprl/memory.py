import random

from collections import namedtuple


Experience = namedtuple(
    'Experience',
    'observation, action, reward, next_observation, done'
)


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in xrange(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.length += 1
        elif self.length == self.maxlen:
            self.start = (self.start + 1) % self.maxlen
        else:
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class Memory(object):
    def __init__(self, limit):
        self.limit = limit
        self.experiences = RingBuffer(self.limit)

    def __len__(self):
        return len(self.experiences)

    def reset_memory(self):
        self.experiences = RingBuffer(self.limit)

    def remember(self, observation, action, reward, next_observation, done):
        self.experiences.append(
            Experience(observation, action, reward, next_observation, done)
        )

    def sample(self, batch_size):
        if len(self) == 0:
            raise ValueError("Memory is empty")

        if batch_size > len(self):
            # sample with replacement
            batch_idxs = np.random.random_integers(0, len(self), size=batch_size)
        else:
            # sample without replacement
            batch_idxs = random.sample(xrange(len(self)), batch_size)

        return [self.experiences[i] for i in batch_idxs]

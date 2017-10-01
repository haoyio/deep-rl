import numpy as np
import random

from collections import namedtuple


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = []

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            self.data.append(v)
            self.length += 1
        elif self.length == self.maxlen:
            self.data[(self.start + self.length - 1) % self.maxlen] = v
            self.start = (self.start + 1) % self.maxlen
        else:
            raise RuntimeError()


class Memory(object):
    def __init__(self, limit):
        self.limit = limit
        self.experiences = RingBuffer(self.limit)

    def __len__(self):
        return len(self.experiences)

    def reset_memory(self):
        self.experiences = RingBuffer(self.limit)

    def remember(self, observation, action, reward, next_observation, done):
        self.experiences.append((observation, action, reward, next_observation, done))

    def sample(self, batch_size):
        if len(self) == 0:
            raise ValueError("Memory is empty")
        return random.sample(self.experiences.data, batch_size)

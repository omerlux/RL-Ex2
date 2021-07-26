import random
from collections import namedtuple

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'not_done'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        # TODO - part b
        # filling the buffer to capacity
        if len(self.memory) < self.capacity:
            self.memory.append(args)
        else:       # full buffer - run over old elements
            self.memory[self.position] = args
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # TODO - part b
        return random.sample(self.memory, min(len(self.memory), batch_size))        # to avoid bigger batch than buffer

    def __len__(self):
        return len(self.memory)

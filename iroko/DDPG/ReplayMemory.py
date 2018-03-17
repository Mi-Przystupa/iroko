import random

class ReplayMemory:
    def __init__(self, size = 40, dims = 41):
        self.memory = []
        self.indx = 0
        self.size = int(size)

    def push(self, s, a, r, sprime):
        mem = {'s': s, 'a': a, 'r': r, 'sprime': sprime}
        if(len(self.memory) < self.size):
            self.memory.append(mem)
        else:
            self.memory[self.indx] = mem
        self.indx = (self.indx + 1) % self.size
    def batch(self, batchSize):
        return random.sample(self.memory, batchSize)

    def singleSample(self):
        return random.sample(self.memory, 1)

    def len(self):
        return len(self.memory)

    def isFull(self):
        return len(self.memory) == self.size

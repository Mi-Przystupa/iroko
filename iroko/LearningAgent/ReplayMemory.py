import random
import numpy as np

class ReplayMemory:
    def __init__(self, size = 40, dims = 41):
        try:
            self.memory = np.load('buffer.npy').tolist()
            self.indx = np.load('index.npy').item()
        except (IOError,EOFError) as e:
            self.indx = 0
            self.memory = [] 
        self.size = int(size)
        self.toSave = 0
        #if the requested buffer size is smaller than the requested buffer, sample self.size of buffer
        # and set index so that it is still in range
        if(len(self.memory) > self.size):
            self.memory = random.sample(self.memory, self.size)
            self.indx = self.indx % self.size


    def push(self, s, a, r, sprime):
        mem = {'s': s, 'a': a, 'r': r, 'sprime': sprime}
        if(len(self.memory) < self.size):
            self.memory.append(mem)
        else:
            self.memory[self.indx] = mem
        self.indx = (self.indx + 1) % self.size
        if (self.toSave > 8):
            self.toSave = 0
#            np.save('buffer', self.memory)
#            np.save('index', self.indx)
        else:
            self.toSave += 1
    def batch(self, batchSize):
        return random.sample(self.memory, batchSize)

    def singleSample(self):
        return random.sample(self.memory, 1)

    def len(self):
        return len(self.memory)

    def isFull(self):
        return len(self.memory) > 128 #len(self.memory) == self.size

import numpy 
import torch
import random

class CircularList:
    def __init__(self, capacity = 15):
        self.memory = []
        self.indx = 0
        self.capacity = capacity
    def push(self, data):
        if (len(self.memory) < self.capacity):
            self.memory.append(None)
        self.memory[self.indx] = data
        self.indx = (self.indx + 1) % self.capacity

    def sample(self):
        if(len(self.memory) > 0):
            return random.sample(self.memory, 1)
        else:
            return 0

    def __len__(self):
        return len(self.memory)

class LearningAgent:
    def __init__(self, capacity= 15):
        self.memory = CircularList(capacity)


    def addMemory(self, data):
        self.memory.push(data)

    def getSample(self):
        return self.memory.sample()



"""
hello 


"""

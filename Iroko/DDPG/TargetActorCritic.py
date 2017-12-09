from Actor import Actor
from Critic import Critic
from ReplayMemory import ReplayMemory
import torch
from torch.autograd import Variable
import torch.optim as optim
import copy
import random
class TargetActorCritic:
    def __init__(self, actor, critic, memory, s, a, tau, epsilon=0.5):
        self.memory = ReplayMemory(memory)
        self.targetActor = copy.deepcopy(actor) 
        self.targetCritic = copy.deepcopy(critic) 
        self.tau = tau
        self.epsilon = epsilon
        #more a dimensionality thing
        self.state = s
        self.action = a 

    def processNoise(self):
        #this should be something more eloquent....
        ret = torch.rand(self.action) / 2.0 
        for i in range(0, self.action):
            r = random.random()
            if ( r <= .33):
                ret[i] = ret[i]
            elif ( .33 < r and r <=.66):
                ret[i] = 0
            else:
                ret[i] = -ret[i]
        return ret 

    def selectAction(self, state):
        #remember, state better be an autograd Variable
        ret = self.targetActor(Variable(state)).data
        if(random.random() > self.epsilon):
            ret = ret + self.processNoise()

        return torch.clamp(ret, 0.0, 1.0) 

    def addToMemory(self, state, action, reward, stateprime):
        self.memory.push(state, action, reward, stateprime)

    def getBatchMemory(self, batchsize):
        return self.memory.batch(batchsize)

    def primedToLearn(self):
        return self.memory.isFull()

    def UpdateTargetNetworks(self, critic, actor): 
        criticDict = critic.state_dict()
        tCriticDict = self.targetCritic.state_dict()
        for param in criticDict.keys():
            tCriticDict[param] = tCriticDict[param] * (1 - self.tau) + criticDict[param] * self.tau

        actorDict = actor.state_dict()
        tActorDict = self.targetActor.state_dict()
        for param in actorDict.keys():
            tActorDict[param] = tActorDict[param] * (1 - self.tau) + actorDict[param] * self. tau

        self.targetCritic.load_state_dict(tCriticDict)
        self.targetActor.load_state_dict(tActorDict)




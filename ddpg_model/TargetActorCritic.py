from Actor import Actor
from Critic import Critic
from ReplayMemory import ReplayMemory
import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
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
        self.OUarray = np.zeros((1000, self.action),dtype="f")
        self.step = 0

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

    def OUprocess(self, sigma, theta, mu):
        # define model parameters
        t_0 = 0
        t_end = 10
        length = 1000

        y = np.zeros((length, self.action),dtype="f")
        t = np.linspace(t_0,t_end,length) # define time axis
        dt = np.mean(np.diff(t))
        drift = lambda y,t: theta*(mu-y) # define drift term
        diffusion = lambda y,t: sigma # define diffusion term

        # solve SDE
        for j in xrange(1, self.action):
            y[0][j] = np.random.normal(loc=0.0,scale=1.0) # initial condition
            noise = np.random.normal(loc=0.0,scale=1.0,size=length)*np.sqrt(dt) #define noise process
            for i in xrange(1,length):
                y[i][j] = y[i-1][j] + drift(y[i-1][j],i*dt)*dt + diffusion(y[i-1][j],i*dt)*noise[i]
        self.OUarray = y


    def selectAction(self, state):
        #remember, state better be an autograd Variable
        ret = self.targetActor(Variable(state)).data
        ret = ret + torch.from_numpy(self.OUarray[self.step]) 

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




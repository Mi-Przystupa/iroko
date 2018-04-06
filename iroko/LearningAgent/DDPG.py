from ReplayMemory import ReplayMemory
from OUNoise import OUNoise
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class Actor(nn.Module):
    def __init__(self, state = 54, actions = 18,  hidden1=400 , hidden2= 300 ,  convNet=None):
        super(Actor, self).__init__()

        if convNet:
            self.convNet = copy.deepcopy(convNet)
        else:
            self.convNet = convNet
        self.normalize = nn.BatchNorm1d(state)
        self.normalize2 = nn.BatchNorm1d(hidden1)
        self.hidden1 = nn.Linear(state, hidden1)  
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.outputs = nn.Linear(hidden2, actions)

        nn.init.xavier_normal(self.hidden1.weight.data, gain=2)
        nn.init.xavier_normal(self.hidden2.weight.data, gain=2)
        nn.init.uniform(self.outputs.weight.data, -1e-3, 1e-3)


    def forward(self,  state):
        x = state
        print(x)
        if self.convNet:
            x = self.convNet(x)
            x = x.view(x.size()[0], -1)

        x = self.normalize(x)
        x = F.relu(self.hidden1(x))
        x = self.normalize2(x)
        x = F.relu(self.hidden2(x))
        return F.sigmoid( self.outputs(x))

class Critic(nn.Module):
    def __init__(self, state = 54, actions = 18, hidden1=400, hidden2 = 300, convNet=None):
        super(Critic, self).__init__()

        if convNet:
            self.convNet = copy.deepcopy(convNet)
        else:
            self.convNet = convNet
        #paper claims 1st layer they do not put action through 1st layer
        self.normalize = nn.BatchNorm1d(state)
        self.normalize2 = nn.BatchNorm1d(hidden1 + actions)
        self.hidden1 = nn.Linear(state, hidden1)
        self.hidden2 = nn.Linear(hidden1 + actions, hidden2)
        self.outputs = nn.Linear(hidden2, 1)

        nn.init.xavier_normal(self.hidden1.weight.data, gain=2)
        nn.init.xavier_normal(self.hidden2.weight.data, gain=2)
        nn.init.uniform(self.outputs.weight.data, -1e-3, 1e-3)

    def forward(self, state, action):
        x = state
        print(x)
        if self.convNet:
            x = self.convNet(x)
            x = x.view(x.size()[0], -1)

        x = self.normalize(x)
        x = F.relu(self.hidden1(x))
        x = torch.cat((x, action),dim=-1)
        #x = self.normalize2(torch.cat(x, action))
        x = F.relu(self.hidden2(x))
        return self.outputs(x)

class DDPG:
    def __init__(self, s, a, tau=.001, gamma=.99, memory=1e3, learningRate = 1e-3,criticpath=None, actorpath=None, \
            h1=400, h2=300, dims=None):
        #include dims if you want to convolve
        if dims:
            convNet = self._createConvolutionNet(dims['c'])
        else:
            convNet = None

        self.gamma =gamma
        self.memory = ReplayMemory(memory)
        self.actor = Actor(state= s, actions = a,hidden1=h1, hidden2=h2,convNet=convNet)
        self.critic = Critic(state = s, actions = a, hidden1=h1, hidden2=h2, convNet=convNet)
        if(not(criticpath== None)):
            try:
                self.critic.load_state_dict(torch.load(criticpath))
            except RuntimeError, e:
                print('Failed to load requested Critic: {}'.format(str(e)))
            except KeyError, e:
                print('Failed to load requested Critic: {}'.format(str(e)))
            except (IOError, EOFError) as e:
                print('Failed to load requested Critic: {}'.format(str(e)))
                
        if(not(actorpath==None)):
            try:
                self.actor.load_state_dict(torch.load(actorpath))
            except RuntimeError, e:
                print('Failed to load requested Actor: {}'.format(str(e)))
            except KeyError, e:
                print('Failed to load requested Actor: {}'.format(str(e)))
            except (IOError, EOFError) as e:
                print('Failed to load requested Actor: {}'.format(str(e)))
        self.targetActor = Actor(state= s, actions = a,hidden1=h1, hidden2=h2,convNet=convNet)
        self.targetActor.load_state_dict(self.actor.state_dict())
        self.targetCritic = Critic(state= s, actions = a,hidden1=h1, hidden2=h2, convNet=convNet)
        self.targetCritic.load_state_dict(self.critic.state_dict())
        self.tau = tau

        self.actorOptimizer = optim.Adam(self.actor.parameters(),learningRate * 1e-1)
        self.criticOptimizer = optim.Adam(self.critic.parameters(),learningRate, weight_decay=1e-2)
        #more a dimensionality thing
        self.state = s
        self.action = a
        if(dims):
            assert(type(dims) is dict)
            assert(len(dims) == 3)
        self.dims = dims
        self.process = OUNoise(a, scale=1.0, mu=0, theta=.15, sigma=0.2)
        self.isExplore = True
        #self.step = 0

    def _createConvolutionNet(self, frames):
        return torch.nn.Sequential(
                torch.nn.BatchNorm2d(frames),
                torch.nn.Conv2d(frames, 32, 3,stride=(2,1),padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.Conv2d(32, 32, 3, stride=(2,1), padding=1),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(32),
                torch.nn.Conv2d(32, 32, 3, stride=(2,1), padding=1),
                torch.nn.ReLU()
                )



    def setExploration(self, scale, sigma, theta, mu):
        self.OUProcess = OUNoise(self.action, scale, sigma, theta,mu)
    def explore(self):
        self.isExplore = True
    def exploit(self):
        self.isExplore = False

    def selectAction(self, state):
        #remember, state better be an autograd Variable
        self.targetActor.eval()
        ret = self.targetActor(Variable(state)).data
        self.targetActor.train()
        if (self.isExplore):
            ret = ret + torch.from_numpy(self.process.noise()).float()
        #self.step += 1
        return ret

    def addToMemory(self, state, action, reward, stateprime):
        self.memory.push(state, action, reward, stateprime)
    def primedToLearn(self):
        return self.memory.isFull()

    def PerformUpdate(self,batchsize):
        if self.dims:
            states = torch.zeros(batchsize, self.dims['c'], self.dims['h'], self.dims['w'])
            statesP = torch.zeros(batchsize,self.dims['c'], self.dims['h'], self.dims['w'])
        else:
            states = torch.zeros(batchsize, self.state)
            statesP = torch.zeros(batchsize, self.state)

        actions = torch.zeros(batchsize, self.action)
        rewards = torch.zeros(batchsize, 1)
         
        for i, sample in enumerate(self.memory.batch(batchsize)):
            actions[i] = sample['a']
            states[i] = sample['s']
            rewards[i] = sample['r']
            statesP[i] = sample['sprime']

        #critic update
        self.criticOptimizer.zero_grad()
        targets = self.targetCritic(Variable(statesP), self.targetActor(Variable(statesP)).detach()).detach()
        y = Variable(rewards + self.gamma * targets.data).detach()
        Q = self.critic(Variable(states), Variable(actions))
        criterion = torch.nn.MSELoss()
        loss = criterion(Q, y)
        loss.backward()
        self.criticOptimizer.step()
        
        #actor update
        self.actorOptimizer.zero_grad()
        A = self.actor(Variable(states, requires_grad=True))
        J = self.critic(Variable(states), A ) 
        loss = -torch.mean(J)#J.mean() #-torch.sum(Q)#backward()
        loss.backward()
        self.actorOptimizer.step()
        
    def UpdateTargetNetworks(self):
        criticDict = self.critic.state_dict()
        tCriticDict = self.targetCritic.state_dict()
        for param in criticDict.keys():
            tCriticDict[param] = tCriticDict[param] * (1 - self.tau) + criticDict[param] * self.tau
        self.targetCritic.load_state_dict(tCriticDict)
        actorDict = self.actor.state_dict()
        tActorDict = self.targetActor.state_dict()
        for param in actorDict.keys():
            tActorDict[param] = tActorDict[param] * (1 - self.tau) + actorDict[param] * self. tau

        self.targetActor.load_state_dict(tActorDict)
    def saveActorCritic(self):
        torch.save(self.critic.state_dict(), 'critic')
        torch.save(self.actor.state_dict(), 'actor')

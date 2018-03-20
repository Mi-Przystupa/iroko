from ReplayMemory import ReplayMemory
from OUNoise import OUNoise
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class Actor(nn.Module):
    def __init__(self, state = 54, actions = 18,  hidden1=400 , hidden2= 300 , useSigmoid=False):
        super(Actor, self).__init__()
        self.useSigmoid=useSigmoid
        self.convNorm = nn.BatchNorm2d(3)
        self.convNorm2= nn.BatchNorm2d(32)
        self.convNorm3 = nn.BatchNorm2d(32)
        self.normalize = nn.BatchNorm1d(state)
        self.normalize2 = nn.BatchNorm1d(hidden1)
        
        self.conv1 = nn.Conv2d(3, 32, 3,stride=(2,1))
        self.conv2 = nn.Conv2d(32,32, 3, stride=(2,1))
        self.conv3 = nn.Conv2d(32, 32, 3, stride=(2,1))

        self.hidden1 = nn.Linear(state, hidden1)  
        self.hidden2 = nn.Linear(hidden1, hidden2)
        self.outputs = nn.Linear(hidden2, actions)

        nn.init.xavier_normal(self.hidden1.weight.data, gain=2)
        nn.init.xavier_normal(self.hidden2.weight.data, gain=2)
        nn.init.uniform(self.outputs.weight.data, -3e-3, 3e3)


    def forward(self,  state):
        # conv layers
        x = self.convNorm(state)
        x = self.conv1(x)
        x = self.convNorm2(F.relu(x))
        x = self.conv2(x)
        x = self.convNorm3(F.relu(x))
        #x = self.conv3(x)
        x = x.view(x.size()[0], -1)

        #fully connected layers
        x = self.normalize(x)
        x = F.relu(self.hidden1(x))
        x = self.normalize2(x)
        x = F.relu(self.hidden2(x))
        if (self.useSigmoid):
            return F.sigmoid(self.outputs(x))
        else:
            return F.tanh( self.outputs(x))

class Critic(nn.Module):
    def __init__(self, state = 54, actions = 18, hidden1=400, hidden2 = 300):
        super(Critic, self).__init__()
        self.convNorm = nn.BatchNorm2d(3)
        self.convNorm2= nn.BatchNorm2d(32)
        self.convNorm3 = nn.BatchNorm2d(32)
        self.normalize = nn.BatchNorm1d(state)
        self.normalize2 = nn.BatchNorm1d(hidden1)
        
        self.conv1 = nn.Conv2d(3, 32, 3,stride=(2,1))
        self.conv2 = nn.Conv2d(32, 32, 3, stride=(2,1))
        self.conv3 = nn.Conv2d(32, 32, 3, stride=(2,1))

        #paper claims 1st layer they do not put action through 1st layer
        self.hidden1 = nn.Linear(state, hidden1)
        self.hidden2 = nn.Linear(hidden1 + actions, hidden2)
        self.outputs = nn.Linear(hidden2, 1)

        nn.init.xavier_normal(self.hidden1.weight.data, gain=2)
        nn.init.xavier_normal(self.hidden2.weight.data, gain=2)
        nn.init.uniform(self.outputs.weight.data, -3e-3, 3e3)



    def forward(self, state, action):
        x = self.convNorm(state)
        x = self.conv1(x)
        x = self.convNorm2(F.relu(x))
        x = self.conv2(x)
        x = self.convNorm3(F.relu(x))
        #x = self.conv3(x)
        x = x.view(x.size()[0], -1)

        #Fully connected layers
        x = self.normalize(x)
        x = F.relu(self.hidden1(x))
        x = torch.cat((x, action),dim=-1)
        #x = self.normalize2(torch.cat(x, action))
        x = F.relu(self.hidden2(x))
        return self.outputs(x)

class DDPGConv:
    def __init__(self, gamma, memory, s, a, tau, learningRate = 1e-3,criticpath=None, actorpath=None, useSig=False, h1=400, h2=300, h=5, w=16):
        self.gamma =gamma
        self.memory = ReplayMemory(memory)
        self.height = h
        self.width = w
        self.actor = Actor(state= s, actions = a,hidden1=h1, hidden2=h2,  useSigmoid=useSig)
        self.critic = Critic(state = s, actions = a, hidden1=h1, hidden2=h2)
        if(not(criticpath== None)):
            try:
                self.critic.load_state_dict(torch.load(criticpath))
            except RuntimeError, e:
                print('Failed to load requested Critic: {}'.format(str(e)))
            except KeyError, e:
                print('Failed to load requested Critic: {}'.format(str(e)))
            except IOError, e:
                print('Failed to load requested Critic: {}'.format(str(e)))
                
        if(not(actorpath==None)):
            try:
                self.actor.load_state_dict(torch.load(actorpath))
            except RuntimeError, e:
                print('Failed to load requested Actor: {}'.format(str(e)))
            except KeyError, e:
                print('Failed to load requested Actor: {}'.format(str(e)))
            except IOError, e:
                print('Failed to load requested Actor: {}'.format(str(e)))
        self.targetActor = Actor(state= s, actions = a,hidden1=h1, hidden2=h2, useSigmoid=useSig)
        self.targetActor.load_state_dict(self.actor.state_dict())
        self.targetCritic = Critic(state= s, actions = a,hidden1=h1, hidden2=h2)
        self.targetCritic.load_state_dict(self.critic.state_dict())
        self.tau = tau

        self.actorOptimizer = optim.Adam(self.actor.parameters(),learningRate * 1e-1)
        self.criticOptimizer = optim.Adam(self.critic.parameters(),learningRate, weight_decay=1e-2)
        #more a dimensionality thing
        self.state = s
        self.action = a
        #self.OUarray = np.zeros((1000, self.action),dtype="f")
        self.OUProcess = OUNoise(a,scale=1.0, sigma=0.2, theta=.15, mu=0.0)
        self.isExplore = True
        #self.step = 0

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
            ret = ret + torch.from_numpy(self.OUProcess.noise()).float()
        #self.step += 1
        return ret

    def addToMemory(self, state, action, reward, stateprime):
        self.memory.push(state, action, reward, stateprime)
    def primedToLearn(self):
        return self.memory.isFull()

    def PerformUpdate(self,batchsize):
        actions = torch.zeros(batchsize, self.action)
        states = torch.zeros(batchsize, 3, self.height, self.width)
        rewards = torch.zeros(batchsize, 1)
        statesP = torch.zeros(batchsize, 3, self.height,  self.width) 
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
        A = self.actor(Variable(states))
        J = -self.critic(Variable(states), A ) 
        J = J.mean() #-torch.sum(Q)#backward()
        J.backward()
        self.actorOptimizer.step()
        

    def PerformUpdateOld(self, batchsize):
        #Mildly important, according to https://github.com/vy007vikas/PyTorch-ActorCriticRL
        # the criterion on the actor is this: sum(-Q(s,a)) I'm assuming this is over the batch....
        self.actorOptimizer.zero_grad()
        self.criticOptimizer.zero_grad()

        batch = self.memory.batch(batchsize)
        Q = torch.zeros(len(batch),self.state + self.action )
        Qprime = torch.zeros(len(batch),self.state + self.action )
        rewards = torch.zeros(len(batch), 1)
        # This loop should generate all Q values for the batch
        i = 0
        for sample in batch:
            Q[i,:]= torch.cat((sample['s'], sample['a']))
            transition = self.targetActor(Variable(sample['sprime'],volatile=True)).data
            Qprime[i,:]  = torch.cat((sample['sprime'], transition),dim=0)
            rewards[i,0] = sample['r'][0]
            i += 1

        #Critic Update
        Qprime = self.gamma * self.targetCritic(Variable(Qprime)).data + rewards
        Qprime = Variable(Qprime)
        Q = self.critic(Variable(Q))
        criterion = torch.nn.MSELoss()
        loss = criterion(Q, Qprime)
        loss.backward()
        self.criticOptimizer.step()

        criterion = torch.nn.MSELoss()

        self.actorOptimizer.zero_grad()
        S = torch.zeros(len(batch), self.state)
        i = 0
        for sample in batch:
            S[i,:]= sample['s']
            i += 1
        A = self.actor(Variable(S))
        loss = -1 *self.critic(Variable(S),A).mean() #-1 * torch.sum(self.critic(torch.cat((Variable(S),A), dim=1)))
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

from Actor import Actor
from Critic import Critic
from ReplayMemory import ReplayMemory
from OUNoise import OUNoise
import torch
from torch.autograd import Variable
import torch.optim as optim
import random
import numpy as np
class DDPG:
    def __init__(self, gamma, memory, s, a, tau, learningRate = 1e-3,criticpath=None, actorpath=None, useSig=False):
        self.gamma =gamma
        self.memory = ReplayMemory(memory)
        self.actor = Actor(state= s, actions = a, useSigmoid=useSig)
        self.critic = Critic(state = s, actions = a)
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
        self.targetActor = Actor(state= s, actions = a, useSigmoid=useSig)
        self.targetActor.load_state_dict(self.actor.state_dict())
        self.targetCritic = Critic(state= s, actions = a)
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

#    def OUprocess(self, sigma, theta, mu, turnOff=False):
#        # define model parameters
#        t_0 = 0
#        t_end = 10
#        length = 1000
#
#        y = np.zeros((length, self.action),dtype="f")
#        t = np.linspace(t_0,t_end,length) # define time axis
#        dt = np.mean(np.diff(t))
#        drift = lambda y,t: theta*(mu-y) # define drift term
#        diffusion = lambda y,t: sigma # define diffusion term
#
#        # solve SDE
#        for j in xrange(0, self.action):
#            y[0][j] = np.random.normal(loc=0.0,scale=1.0) # initial condition
#            noise = np.random.normal(loc=0.0,scale=1.0,size=length)*np.sqrt(dt) #define noise process
#            for i in xrange(1,length):
#                y[i][j] = y[i-1][j] + drift(y[i-1][j],i*dt)*dt + diffusion(y[i-1][j],i*dt)*noise[i]
#        self.OUarray = y
#        if(turnOff):
#            self.OUarray = np.zeros((length, self.action), dtype="f")

    def selectAction(self, state, toExplore=True):
        #remember, state better be an autograd Variable
        ret = self.targetActor(Variable(state)).data
        if (toExplore):
            ret = ret + torch.from_numpy(self.OUProcess.noise()).float()
        #self.step += 1
        return ret

    def addToMemory(self, state, action, reward, stateprime):
        self.memory.push(state, action, reward, stateprime)
    def primedToLearn(self):
        return self.memory.isFull()

    def PerformUpdate(self,batchsize):
        actions = torch.zeros(batchsize, self.action)
        states = torch.zeros(batchsize, self.state)
        rewards = torch.zeros(batchsize)
        statesP = torch.zeros(batchsize, self.state) 
        for i, sample in enumerate(self.memory.batch(batchsize)):
            actions[i] = sample['a']
            states[i] = sample['s']
            rewards[i] = sample['r']
            statesP = sample['sprime']

        #critic update
        self.criticOptimizer.zero_grad()
        targets = self.targetCritic(Variable(statesP), self.targetActor(Variable(statesP)).detach()).detach()
        y = Variable(rewards + self.gamma * targets.data, volatile=True).detach()
        Q = self.critic(Variable(states), Variable(actions))
        criterion = torch.nn.MSELoss()
        loss = criterion(Q, y)
        loss.backward()
        self.criticOptimizer.step()
        
        #actor update
        self.actorOptimizer.zero_grad()
        A = self.actor(Variable(states))
        Q = -self.critic(Variable(states), A ) 
        Q = Q.mean() #-torch.sum(Q)#backward()
        Q.backward()
        #Q.backward()
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
        torch.save(self.critic.state_dict(), './critic')
        torch.save(self.actor.state_dict(), './actor')

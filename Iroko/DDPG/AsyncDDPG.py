from Actor import Actor
from Critic import Critic
from TargetActorCritic import TargetActorCritic

import torch
from torch.autograd import Variable
import torch.optim as optim
import random
import copy
class AsyncDDPG(object):
    def __init__(self, gamma, s, a, learningRate = 1e-3,criticpath=None, actorpath=None):
        self.gamma =gamma
        self.actor = Actor(state= s, actions = a, hidden1 = 180, hidden2 = 87)
        self.critic = Critic(state = s, actions = a, hidden1 =250, hidden2 = 100)
        if(not(criticpath== None)):
            self.critic.load_state_dict(torch.load(criticpath))
        if(not(actorpath==None)):
            self.actor.load_state_dict(torch.load(actorpath))
        
        self.actorOptimizer = optim.Adam(self.actor.parameters(),learningRate)
        self.criticOptimizer = optim.Adam(self.critic.parameters(),learningRate)
        #more a dimensionality thing
        self.state = s
        self.action = a 
        self.count = 0

    def PerformUpdate(self, batchsize, target):
        #Mildly important, according to https://github.com/vy007vikas/PyTorch-ActorCriticRL
        # the criterion on the actor is this: sum(-Q(s,a)) I'm assuming this is over the batch....
        self.actorOptimizer.zero_grad() 
        self.criticOptimizer.zero_grad()
 
        batch = target.getBatchMemory(batchsize) 

        Q = torch.zeros(len(batch),self.state + self.action )
        Qprime = torch.zeros(len(batch),self.state + self.action )
        rewards = torch.zeros(len(batch), 1)

        # This loop should generate all Q values for the batch
        i = 0
        for sample in batch:
            Q[i,:]= torch.cat((sample['s'], sample['a']))
            transition = target.targetActor(Variable(sample['sprime'],volatile=True)).data
            Qprime[i,:]  = torch.cat((sample['sprime'], transition),dim=0)
            rewards[i,0] = sample['r'][0]
            i += 1

        #Critic Update
        Qprime = self.gamma * target.targetCritic(Variable(Qprime)).data + rewards
        Qprime = Variable(Qprime)

        Q = self.critic(Variable(Q))
        criterion = torch.nn.MSELoss()
        loss = criterion(Q, Qprime)
        loss.backward()
        self.criticOptimizer.step()
 
        criterion = torch.nn.MSELoss()
        #criticupdate
        self.actorOptimizer.zero_grad() 
        S = torch.zeros(len(batch), self.state)
        i = 0
        for sample in batch:
            S[i,:]= sample['s']
            i += 1
        A = self.actor(Variable(S)) 
        loss = -1 * torch.sum(self.critic(torch.cat((Variable(S),A), dim=1)))
        loss.backward()
        self.actorOptimizer.step()
        
    def getActor(self):
        return self.actor

    def getCritic(self):
        return self.critic


    def ProduceTargetActorCritic(self, memory=2000, tau=.25, epsilon=.5 ):
        print(self.count)
        self.count += 1
        s = self.state
        a = self.action
        return TargetActorCritic(self.actor, self.critic, memory, s, a, tau, epsilon=0.5)


    def saveActorCritic(self):
        torch.save(self.critic.state_dict(), './AsyncCritic')
        torch.save(self.actor.state_dict(), './AsyncActor')



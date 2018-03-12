from Actor import Actor
from Critic import Critic
from ReplayMemory import ReplayMemory
import torch
from torch.autograd import Variable
import torch.optim as optim
import random
class DDPG:
    def __init__(self, gamma, memory, s, a, tau, learningRate = 1e-6,criticpath=None, actorpath=None):
        self.gamma =gamma
        self.memory = ReplayMemory(memory)
        self.actor = Actor(state= s, actions = a)
        self.critic = Critic(state = s, actions = a)
        if(not(criticpath== None)):
            self.critic.load_state_dict(torch.load(criticpath))
        if(not(actorpath==None)):
            self.actor.load_state_dict(torch.load(actorpath))
        self.targetActor = Actor(state= s, actions = a)
        self.targetActor.load_state_dict(self.actor.state_dict())
        self.targetCritic = Critic(state= s, actions = a)
        self.targetCritic.load_state_dict(self.critic.state_dict())
        self.tau = tau
        
        self.actorOptimizer = optim.Adam(self.actor.parameters(),learningRate)
        self.criticOptimizer = optim.Adam(self.critic.parameters(),learningRate)
        #more a dimensionality thing
        self.state = s
        self.action = a 

    def processNoise(self):
        #this should be something more eloquent....
        ret = torch.rand(self.action) 
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
        if(random.random() > .5):
            ret = ret + self.processNoise()
        return torch.clamp(ret, -0.9, 0.9) 

    def addToMemory(self, state, action, reward, stateprime):
        self.memory.push(state, action, reward, stateprime)
    def primedToLearn(self):
        return self.memory.isFull()

    def PerformUpdate(self, batchsize):
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
        loss = -1 * torch.sum(self.critic(torch.cat((Variable(S),A), dim=1)))
        loss.backward()
        self.actorOptimizer.step()




    def UpdateTargetNetworks(self): 
        criticDict = self.critic.state_dict()
        tCriticDict = self.targetCritic.state_dict()
        for param in criticDict.keys():
            tCriticDict[param] = tCriticDict[param] * (1 - self.tau) + criticDict[param] * self.tau

        actorDict = self.actor.state_dict()
        tActorDict = self.targetActor.state_dict()
        for param in actorDict.keys():
            tActorDict[param] = tActorDict[param] * (1 - self.tau) + actorDict[param] * self. tau

        self.targetCritic.load_state_dict(tCriticDict)
        self.targetActor.load_state_dict(tActorDict)
    def saveActorCritic(self):
        torch.save(self.critic.state_dict(), './critic')
        torch.save(self.actor.state_dict(), './actor')



from DDPG import DDPG 
import torch.nn 
from torch.autograd import Variable
import torch
import gym
import time
from OUNoise import OUNoise
#(self, inputs, actions, numNeuron1, numNeuron2, alpha, gamma, epsilon = .9, decay=.001):
numSimulation = 800

simulationLengths = torch.zeros(numSimulation);
simulationCount = 0
maxSeen = 0
env = gym.make('MountainCarContinuous-v0')
#env = gym.make('Pendulum-v0')
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = env.action_space.high[0]
A_MIN = env.action_space.low[0]
print('states: {} actions:{}'.format(S_DIM, A_DIM))
print('max: {} min: {}'.format(A_MAX, A_MIN))
#Agent = LearningController(4, 2,8 , 15, .000001, .9, epsilon = .7, decay = 1e-8 )

Agent = DDPG(S_DIM, A_DIM, criticpath='critic', actorpath='actor')
sigma = .2#.3 #.3- makes it between -.5 - .5
theta = .15#.2 #.2
mu = 0.0
scale = 1.0

Agent.setExploration(scale, sigma, theta, mu)
observation = env.reset()

poke = torch.from_numpy(observation).float()
pokeAction = Agent.selectAction(poke.unsqueeze(0))
totalreward = 0
displayResult = False
accel = 0.0
prevV = 0.0
t = .001
avgReward = 0
steps = 0
process = OUNoise(1,scale=1.0, sigma=0.2, theta=.15, mu=0.0)

for i in range(1, numSimulation):
    Agent.setExploration(scale, sigma, theta, mu)
    
    #Agent.exploit()
    #do the thing
    inputs = torch.from_numpy(observation).float()
    action = Agent.selectAction(inputs.unsqueeze(0))
    action = action.squeeze()
    observation, reward, done, info = env.step(action.numpy()) 
    steps += 1

    #create memory
    s = inputs.float()
    a = action.double()
    sp = torch.from_numpy(observation).float()
    r = reward 
    totalreward += r
    #add to agents memory and do update as needed 
    Agent.addToMemory(s, a,  r, sp) 

    while (not done):
        steps += 1
        inputs = torch.from_numpy(observation).float().squeeze()
        action = Agent.selectAction(inputs.unsqueeze(0))
        action = action.squeeze()

        observation, reward, done, info = env.step(action.numpy()) 
        #env.render()    
        #create memory
        s = inputs.float()
        a = action.double()
        sp = torch.from_numpy(observation).float()
        accel = s[1] - prevV
        accel = accel / t
        r = reward 
        totalreward += r
        #print(Agent.critic(Variable(poke),Variable(pokeAction)))
        #add to agents memory and do update as needed 
        Agent.addToMemory(s, a,  r, sp) 
        if (Agent.primedToLearn()):
            Agent.PerformUpdate(64)
            Agent.UpdateTargetNetworks()

    if (steps > 2.5e6):
        print('that is enough training')
        break
    avgReward += totalreward
    print('Simulation: {}, totalReward: {}, averageReward: {}'.format(i, totalreward, avgReward / i ))
    totalreward = 0
    
    #if (i % 100 == 0 and Agent.primedToLearn()):
    #    displayResult = True
    #    Agent.exploit()
    #else:
    #    displayResult = False
    #    Agent.explore()
    Agent.saveActorCritic()
    observation = env.reset()


 
Agent.saveActorCritic()


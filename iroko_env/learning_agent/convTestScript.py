from DDPG import DDPG
import torch.nn
from torch.autograd import Variable
import torch
import gym
import time
import numpy as np
from OUNoise import OUNoise
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import torchvision.transforms as T
#(self, inputs, actions, numNeuron1, numNeuron2, alpha, gamma, epsilon = .9, decay=.001):

# screen obtaining code from pytorch tutorial

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
screen_width = 600  # might have to check this
USE_CUDA = True


def get_screen(env):
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    view_width = 320
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    screen = torch.from_numpy(screen)
    screen = resize(screen).unsqueeze(0).float()
    if USE_CUDA:
        screen = screen.cuda()
    return screen


numSimulation = 800

simulationLengths = torch.zeros(numSimulation)
simulationCount = 0
maxSeen = 0
env = gym.make('MountainCarContinuous-v0')  # env = gym.make('Pendulum-v0')
# visualize a frame
# env.reset()
# plt.figure()
# plt.imshow(get_screen(env.env).squeeze(0).permute(1, 2, 0).numpy(),
#        interpolation='none')
# plt.title('Example extracted screen')
# plt.show()

S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = float(env.action_space.high[0])
A_MIN = float(env.action_space.low[0])
print('states: {} actions:{}'.format(S_DIM, A_DIM))
print('max: {} min: {}'.format(A_MAX, A_MIN))

sigma = .2  # .3 #.3- makes it between -.5 - .5
theta = .15  # .2 #.2
mu = 0.0
scale = 1.0
use_conv = {'c': 3, 'h': 40, 'w': 60}

s = 9600
a = 1

Agent = DDPG(s, a, dims=use_conv, h1=80, h2=50, criticpath='critic', actorpath='actor')

Agent.setExploration(scale, sigma, theta, mu)
Agent.enableCuda()
observation = env.reset()

init_inputs = [get_screen(env).squeeze() for i in range(0, 3)]

totalreward = 0
displayResult = False
accel = 0.0
prevV = 0.0
t = .001
avgReward = 0
steps = 0
process = OUNoise(1, scale=1.0, sigma=0.2, theta=.15, mu=0.0)
for i in range(1, numSimulation):
    Agent.setExploration(scale, sigma, theta, mu)

    # Agent.exploit()
    inputs = get_screen(env)

    action = Agent.selectAction(inputs)
    action = action.squeeze()
    if (USE_CUDA):
        _, reward, done, info = env.step(action.cpu().numpy())
    else:
        _, reward, done, info = env.step(action.numpy())
    observation = get_screen(env)
    steps += 1

    # create memory
    s = inputs.float()
    a = action.double()
    sp = observation.float()
    r = reward
    totalreward += r
    # add to agents memory and do update as needed
    Agent.addToMemory(s, a, r, sp)

    while (not done):
        steps += 1
        inputs = observation.float()
        action = Agent.selectAction(inputs)
        action = action.squeeze()

        if (USE_CUDA):
            _, reward, done, info = env.step(action.cpu().numpy())
        else:
            _, reward, done, info = env.step(action.numpy())

        observation = get_screen(env)
        # env.render()
        # create memory
        s = inputs.float()
        a = action.double()
        sp = observation.float()
        r = reward
        totalreward += r

        # add to agents memory and do update as needed
        Agent.addToMemory(s, a, r, sp)
        if (Agent.primedToLearn()):
            Agent.PerformUpdate(64)
            Agent.UpdateTargetNetworks()

    if (steps > 2.5e6):
        print('that is enough training')
        break
    avgReward += totalreward
    print('Simulation: {}, totalReward: {}, averageReward: {}'.format(i, totalreward, avgReward / i))
    totalreward = 0

    # if (i % 100 == 0 and Agent.primedToLearn()):
    #    displayResult = True
    #    Agent.exploit()
    # else:
    #    displayResult = False
    #    Agent.explore()
    Agent.saveActorCritic()
    observation = env.reset()


Agent.saveActorCritic()

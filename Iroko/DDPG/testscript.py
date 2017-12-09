from Actor import Actor
from Critic import Critic
import torch
from torch.autograd import Variable
import torch.nn as nn


actor = Actor()
state = Variable(torch.rand(41))
print(actor)

print("output of network")
action = actor(state)
print(action)

critic = Critic()

stateaction = Variable(torch.cat((state.data,action.data)))

Qval = critic(stateaction)
print(Qval)

from LearningController import LearningController
import torch.nn 
import torch
import gym


#(self, inputs, actions, numNeuron1, numNeuron2, alpha, gamma, epsilon = .9, decay=.001):
numSimulation = 100000000

simulationLengths = torch.zeros(numSimulation);
simulationCount = 0
maxSeen = 0
Agent = LearningController(4, 2,8 , 15, .001, .9, epsilon = .5, decay = 1e-8 )
#Agent = LearningController(4, 2,8 , 15, .001, .5, 1e-8, './modelconfig')
for params in Agent.model.parameters():
    print(params)

env = gym.make('CartPole-v0')
env.reset()
env.render()
numIterations = 0
sinceFailure = 0
for i in range(1, numSimulation):
     
    action = Agent.PerformAction()
    val, a = torch.max(action, 0)
    observation, reward, done, info = env.step(a[0]) 
    env.render()    
    inputs = torch.from_numpy(observation)
    numIterations += 1
    if( done ):
        simulationLengths[simulationCount] = numIterations
        simulationCount += 1
        reward = -100
        if(maxSeen < numIterations):
            maxSeen = numIterations
            print(maxSeen)
        numIterations = 0
        env.reset()
        sinceFailure = 0
    else:
        reward = 1

    if(i % 10000 == 0):
        print("Saving Network, curr iteration: " + str(i))
        Agent.saveNetwork()

    Agent.UpdateValueFunction(inputs.float(), action.float(), reward)
 
torch.save(simulationLengths[0:simulationCount], './simulationCounts')
Agent.saveNetwork()


from LearningController import LearningController
import torch.nn 

Agent = LearningController(10, 10, 10, 15, .1, .9)


inputs = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
action = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
reward = 1;
oddStates = 0
numSimulation = 1000000
for i in range(1, numSimulation):
    
    Agent.UpdateValueFunction(inputs, action.float(), reward)

    action = Agent.PerformAction()
    inputs = inputs + action.float()
    if (int(torch.sum(inputs)) % 2 != 0):
        reward = 1
        oddStates = oddStates + 1
    else:
        reward = -1 
    print(float(oddStates ) / i )

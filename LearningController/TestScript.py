from LearningController import LearningController
import torch.nn 

Agent = LearningController(10, 2, 10, 15, .1, .9)


inputs = torch.Tensor([1,2,3,4,5,6,7,8,9, 0])
action = torch.Tensor([0 , 1])
reward = 1;

for i in range(0, 1000):
    print( Agent.UpdateValueFunction(inputs, action, reward))

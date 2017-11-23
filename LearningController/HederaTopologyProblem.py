import numpy as np
import torch.nn
import torch
from  torch.autograd import Variable
import time

samples = 6631171 - 20  #14973108 - 60
indexes = np.arange(samples);
np.random.shuffle(indexes)
trainset = indexes[0:int(np.floor(samples * .90))]
testset = indexes[int(np.ceil(samples*.90)):-1]

data = np.load('data.npy')
#data = np.random.rand(samples, 15)

data = data[0:samples, :]

trainset = data[trainset,:]
testset = data[testset,:]

## Set-up that sweet sweet neural network learning
network = torch.nn.Sequential(
    torch.nn.Linear(14, 8, True),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 4,True) ,
    torch.nn.ReLU(),
    torch.nn.Linear(4,83),
    torch.nn.LogSoftmax()) #1))

optimizer = torch.optim.SGD(network.parameters(), lr = 1e-7)

#criterion = torch.nn.MSELoss()
criterion = torch.nn.NLLLoss()
#for sample in trainset:
for i in range(0, trainset.size / 15, 32):
    sample = trainset[i:i+33,:] 
    network.zero_grad()
    #label = Variable(torch.from_numpy(np.array([sample[:,0]])).float())
    #features = Variable(torch.from_numpy(sample[:,1:15]).float())
    label = Variable(torch.from_numpy(np.array([sample[:,0]])).long())
    features = Variable(torch.from_numpy(sample[:,1:15]).float())


  #  print(features)
    prediction = network(features)
    loss = criterion(prediction, label.view(label.size()[1]))
    totalLoss = torch.sum(loss.data, 0) / 33
    #print(totalLoss)
    #print('samples: ' + str(i) + ',' + str(i + 32) + ',' + 'average training loss: ' + str(totalLoss[0]) )
    loss.backward()
    optimizer.step()


total_loss = 0

label = Variable(torch.from_numpy(np.array([testset[:,0]])).long())
features = Variable(torch.from_numpy(testset[:,1:15]).float())


prediction = network(features)
loss = criterion(prediction, label.view(label.size()[1]))
total_loss = torch.sum(loss.data ) 

"""
for sample in testset:
    #label = Variable(torch.from_numpy(np.array([sample[0]])).float())
    #features = Variable(torch.from_numpy(sample[1:15]).float())
    label = Variable(torch.from_numpy(np.array([sample[:,0:83]])).float())
    features = Variable(torch.from_numpy(sample[:,83:(83+14)]).float())


    prediction = network(features)
    loss = criterion(prediction, label)
    total_loss += loss
"""

print(total_loss)
print("Average loss: " + str( total_loss / float(testset.size / 15)))


(maxes, yhat) = torch.max(prediction, 1)

print(yhat.size())
err = (torch.sum(torch.eq(yhat, label)).data) / float(testset.size / 15)
classifications = torch.stack((yhat, label.view(label.data.size()[1])), dim=1 )
print(classifications)
print("THe Accuracy: " + str(err))
print("Number of supposed Misclassifications: " + str(torch.sum(yhat != label).data[0]))
tolookat = 0
while tolookat != -1:
    #a = input("Which label,prediction do you want to see? ")
    time.sleep(1)
    tolookat += 1
    prediction = classifications.data[tolookat,0]
    label = classifications.data[tolookat,1]
    print(str(prediction) + ' , ' +  str(label))

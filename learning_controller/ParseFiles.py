import os
import csv
import numpy as np
currLabel = 0 
#data = np.zeros((14973108, 15)) 
#0 - 83 should be 84 labels
numL = 83
data = np.zeros((6631171,numL + 14))
indx = 0
invalid = 0
labels = dict()
nextLabel = 0
for dirname in os.listdir('./results'):
    if( dirname != 'fattree-hedera'):
        continue
    currLabel = currLabel + 1
    print(dirname)
    for dirname2 in os.listdir('./results/' + dirname):
        for filename in os.listdir('./results/' + dirname +  '/' + dirname2):
            with open('./results/' + dirname +  '/' + dirname2 +'/' + filename, 'rb') as csvfile:
                rates = csv.reader(csvfile, delimiter=',')
                # for this file
                for row in rates:
                    features = []
                    exclude = 0
                    label = np.zeros(numL)
                    #for each element in the row
                    for item in row:
                        if(exclude == 0):
                            exclude = exclude + 1
                            continue
                        elif(exclude == 1):
                            exclude = exclude + 1
                            if(not(item in labels)):
                                    labels[item] = nextLabel
                                    nextLabel += 1
                            label[labels[item]] = 1.0
                        else:
                            features.append(item)
                    features = np.concatenate((label, np.array(features)))
                    if(features.size != (numL + 14)):
                        print(row)
                        invalid += 1 
                    else:
                        data[indx, :] = features
                        indx += 1
    print(indx)

print(nextLabel)
print(data.size)
print(invalid)
#np.save('data', data)

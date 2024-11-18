from dataLoader import DataLoader
from neuralNetwork import Network
import random
import math

layerCount=4
layerDimensions=[32,64,16,4]
weightRange=3
testRangeD=100
testRangeL=10
batchSize=200
d=DataLoader(layerCount,layerDimensions,weightRange,testRangeD,testRangeL,batchSize)
d.createTestData()
#d.assignRandomWeights()
testData=d.getTestData()
weights=d.getWeights()
network=Network(layerCount,layerDimensions,weights)



def train():
    zAvg=0
    for n in range(batchSize):
        rng=n
        g=network.backpropagation(testData[rng]["data"],testData[rng]["label"])
        zAvg+=sum([v**2 for v in testData[rng]["label"]])
        absV=0
        gSum=d.getEmptyWeights()

        for i in range(layerCount-1,0,-1):
            for j in range(layerDimensions[i]):
                gSum["bias"+str(i)][j]+=g["biasG"+str(i)][j]
                for k in range(layerDimensions[i-1]):
                    gSum["matrix"+str(i)][j][k]+=g["matrixG"+str(i)][j][k]

    cost=network.popCostAvg()

    for i in range(layerCount-1,0,-1):
            for j in range(layerDimensions[i]):
                absV+=gSum["bias"+str(i)][j]**2
                for k in range(layerDimensions[i-1]):
                    absV+=gSum["matrix"+str(i)][j][k]**2
    for i in range(layerCount-1,0,-1):
            for j in range(layerDimensions[i]):
                gSum["bias"+str(i)][j]*=math.pow(1/absV,2/4)
                for k in range(layerDimensions[i-1]):
                    gSum["matrix"+str(i)][j][k]*=math.pow(1/absV,2/4)
    for i in range(layerCount-1,0,-1):
            for j in range(layerDimensions[i]):
                weights["bias"+str(i)][j]+=-gSum["bias"+str(i)][j]
                for k in range(layerDimensions[i-1]):
                    weights["matrix"+str(i)][j][k]+=-gSum["matrix"+str(i)][j][k]
    d.updateWeights(weights)
    print("absV: "+str(math.sqrt(absV)))
    print(cost)
    print("cost of zero-weights avg: "+str(zAvg/batchSize))
    #zAvg=0
    return cost

def trainBatch():
    a=1000000000
    n=0
    
    d.createTestData()
    newTestData=d.getTestData()
    for data in newTestData:
        testData[n]=data
        n+=1
    n=0
    while a>1000 and n<1:
        a=train()
        n+=1


for i in range(10000):
    print("batch nr: "+str(i+1))
     
    trainBatch()
pass



import json
import random

class DataLoader():
    
    def __init__(self,layerCount,layerDimensions,weightRange,testRangeD,testRangeL,batchSize):
        self.layerCount=layerCount
        self.layerDimensions=layerDimensions
        self.weightRange=weightRange
        self.testRangeD=testRangeD
        self.testRangeL=testRangeL
        self.batchSize=batchSize

    def getTestData(self):
        with open("testData.json") as file:
            data= json.load(file)
        return data
    
    def getWeights(self):
        with open("weights.json") as file:
            data= json.load(file)
        return data
    
    def updateWeights(self,newWeights):
        with open("weights.json","w") as file:
            json.dump(newWeights,file)

    def createTestData(self):
        newTestData=[]
        for i in range(self.batchSize):
            newTestData.append({})
            newData=[]
            newLabel=[]
            for n in range(self.layerDimensions[self.layerCount-1]):
                newLabel.append(random.randint(-1000*self.testRangeL,1000*self.testRangeL)/1000)
            for j in range(int(self.layerDimensions[0]/2)):
                x=random.randint(-1000*self.testRangeD,1000*self.testRangeD)/1000
                y=0
                for k in range(self.layerDimensions[self.layerCount-1]):
                    y+=newLabel[k]*x**k
                newData.append(x)
                newData.append(y)
            newTestData[i]["label"]=newLabel
            newTestData[i]["data"]=newData
        with open("testData.json","w") as file:
            json.dump(newTestData,file)

    def assignRandomWeights(self):
        newWeights={
            "matrix1":[],
            "matrix2":[],
            "matrix3":[],
            "bias1":[],
            "bias2":[],
            "bias3":[]
        }
        for n in range(1,self.layerCount):
            for i in range(self.layerDimensions[n]):
                newWeights["matrix"+str(n)].append([])
                newWeights["bias"+str(n)].append(random.randint(-1000*self.weightRange,1000*self.weightRange)/1000)
                for j in range(self.layerDimensions[n-1]):
                    newWeights["matrix"+str(n)][i].append(random.randint(-1000*self.weightRange,1000*self.weightRange)/1000)
        with open("weights.json","w") as file:
            json.dump(newWeights,file)

    def getEmptyWeights(self):
        newWeights={
            "matrix1":[],
            "matrix2":[],
            "matrix3":[],
            "bias1":[],
            "bias2":[],
            "bias3":[]
        }
        for n in range(1,self.layerCount):
            for i in range(self.layerDimensions[n]):
                newWeights["matrix"+str(n)].append([])
                newWeights["bias"+str(n)].append(0)
                for j in range(self.layerDimensions[n-1]):
                    newWeights["matrix"+str(n)][i].append(0)
        return newWeights

pass

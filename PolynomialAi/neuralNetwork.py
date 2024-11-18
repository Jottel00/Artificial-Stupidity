import math

class Network:
    
    def __init__(self,layerCount,layerDimensions,weights):
        self.layerCount=layerCount
        self.layerDimensions=layerDimensions
        self.weights=weights
        self.costSum=0
        self.trainingsCount=0

    def layerActivation(self,layerIn,matrix,bias,dimIn,dimOut):
        layerOut=[]
        for i in range(dimOut):
            layerOut.append(bias[i])
            for j in range(dimIn):
                layerOut[i]+=layerIn[j]*matrix[i][j]
            if(dimOut!=self.layerDimensions[self.layerCount-1]): layerOut[i]=math.atan(layerOut[i])
        return layerOut

    def evaluate(self,inputData):
        neuronLayers=[]
        neuronLayers.append(inputData)
        for i in range(1,self.layerCount):
            neuronLayers.append(self.layerActivation(neuronLayers[i-1],self.weights["matrix"+str(i)],self.weights["bias"+str(i)],self.layerDimensions[i-1],self.layerDimensions[i]))
        return neuronLayers
    
    def backpropagation(self,inputData,validationData):
        activations=self.evaluate(inputData)
        cost=sum([(activations[self.layerCount-1][n]-validationData[n])**2 for n in range(self.layerDimensions[self.layerCount-1])])
        #print("cost: "+str(cost))
        self.costSum+=cost
        self.trainingsCount+=1
        gradient={}
        absV=0
        actDer=[[2*(activations[self.layerCount-1][n]-validationData[n]) for n in range(self.layerDimensions[self.layerCount-1])]]
        for i in range(self.layerCount-1,0,-1):
            gradient["biasG"+str(i)]=[]
            gradient["matrixG"+str(i)]=[]
            actDer.insert(0,[0 for k in range(self.layerDimensions[i-1])])
            for j in range(self.layerDimensions[i]):
                if(i!=3):gradient["biasG"+str(i)].append(actDer[1][j]*1/(1+math.tan(activations[i][j])**2))
                else:gradient["biasG"+str(i)].append(actDer[1][j])
                #absV+=actDer[1][j]**2
                if(i!=3):gradient["matrixG"+str(i)].append([activations[i-1][k]*actDer[1][j]*1/(1+math.tan(activations[i][j])**2) for k in range(self.layerDimensions[i-1])])
                else:gradient["matrixG"+str(i)].append([activations[i-1][k]*actDer[1][j] for k in range(self.layerDimensions[i-1])])
                #absV+=sum([(activations[i-1][k]*actDer[1][j])**2 for k in range(self.layerDimensions[i-1])])
                for k in range(self.layerDimensions[i-1]):
                    if(i!=3):actDer[0][k]+=self.weights["matrix"+str(i)][j][k]*actDer[1][j]*1/(1+math.tan(activations[i][j])**2)
                    else:actDer[0][k]+=self.weights["matrix"+str(i)][j][k]*actDer[1][j]
        return gradient
    
    def popCostAvg(self):
        tmp=self.costSum/self.trainingsCount
        self.costSum=0
        self.trainingsCount=0
        return tmp
                


        



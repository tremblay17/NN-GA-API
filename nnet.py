import math
import numpy as np
from numpy import maximum, random as rand

class Perceptron:
    def __init__(self, inputs, hiddenLayers, outputLayer,
                    loss='sumSquaredErrors', activation='sigmoid', epochs=25, learningRate=0.1, momentum=0.95):
        '''
        Arguments for Layers:
            inputs = [x1,...,xN]
            hiddenLayer = [numLayers, layer1_size, ..., layerN_size]
            outputLayer = [size]
        '''
        #randomize weights and thresholds for hiddenLayers and outputLayer
        self.outputLayer = outputLayer
        self.hiddenLayers = hiddenLayers[1:]
        self.weights = []
        self.thresholds = []
        nodeWeights = []
        layerThresholds =[]
        for i in range(1,hiddenLayers[0]+1):
            layerThresholds = []
            for j in range(hiddenLayers[i]):
                nodeWeights = []
                if(i==1):
                    self.inputs=len(inputs)
                else:
                    self.inputs=hiddenLayers[i-1]
                layerThresholds.append(rand.uniform(-2.4/float(self.inputs),2.4/float(self.inputs)))
                for k in range(self.inputs):
                    nodeWeights.append(rand.uniform(-2.4/float(self.inputs),2.4/float(self.inputs)))
                self.weights.append(nodeWeights)
            self.thresholds.append(layerThresholds)
        self.inputs = inputs
        
        layerThresholds = []
        nodeWeights = []
        for i in range(self.outputLayer):
            layerThresholds.append(rand.uniform(-2.4/float(len(inputs)),2.4/float(len(inputs))))
            for j in range(hiddenLayers[-1]): #inputs to the output layer
                nodeWeights.append(rand.uniform(-2.4/float(len(inputs)),2.4/float(len(inputs))))
        self.thresholds.append(layerThresholds)

        tmp = []
        index = 0
        for i in range(len(self.hiddenLayers)):
            tmp.append(self.weights[index:index+self.hiddenLayers[i]])
            index += self.hiddenLayers[i]
        self.weights = tmp
        self.weights.append(nodeWeights)

        self.hiddenLayers = np.array(self.hiddenLayers)
        self.outputLayer = np.array(self.outputLayer)
        self.inputs = np.array(self.inputs)
        #self.weights = np.array(self.weights)
        #self.thresholds = np.array(self.thresholds)

        self.loss = loss
        self.activation = activation
        self.epochs = epochs
        self.learningRate = learningRate
        self.momentum = momentum

    def getWeights(self):
        pass
    def getThresholds(self):
        pass
    def getInfo(self):
        '''
        getLayerInfo
        getErrorInfo
        getTrainInfo
        getLossInfo
        '''
        pass
    def getTrainInfo(self):
        pass
    def getLayerInfo(self):
        '''
        Num of Hidden Layers
        Each hidden layer size & weights
        output layer size & weights
        '''
        pass
    def getErrorInfo(self):
        '''
        Avg error
        Avg error gradient
        Latest error 
        Latest error gradient
        '''
        pass
    def getLossInfo(self):
        '''
        Avg Loss
        Latest Loss
        '''
        pass
    def setWeights(self):
        pass
    def setThresholds(self):
        pass

    def __buildOutputMat(self):
        pass
    def __buildWeightMat(self):
        pass
    
    ##TODO: Activation Functions
    def __linear(self, inputs, weights, hiddenLayerNum):
        return self.__summation(inputs,weights)
    def __step(self, inputs, weights, hiddenLayerNum):
        return 1 if self.__summation(inputs,weights)>=0 else 0
    def __sigmoid(self, inputs, weights, hiddenLayerNum):
        return 1/(1+math.e**(-self.__summation(inputs,weights)))
    def __tanh(self, inputs, weights, a=1.716,b=0.667):
        return ((2*a)/(1+math.e**(-b*self.__summation(inputs,weights))))-a
    def __relu(self, inputs, weights, hiddenLayerNum):
        return maximum(0,self.__summation(inputs,weights))
    def __leakyRelu(self, inputs, weights, hiddenLayerNum):
        return maximum(0.01*self.__summation(inputs,weights),self.__summation(inputs,weights))
    def activateNode(self, activation, inputs, weights):
        try:
            match(activation):
                case "linear":
                    self.__linear(inputs, weights)
                case "step":
                    self.__step(inputs, weights)
                case "sigmoid":
                    self.__sigmoid(inputs, weights)
                case "tanh":
                    self.__tanh(inputs, weights)
                case "relu":
                    self.__relu(inputs, weights)
                case "leakyRelu":
                    self.__leakyRelu(inputs, weights)
                case _:
                    raise NotImplementedError
        except NotImplementedError:
            print("Error: activation function not implemented: ", activation)
            return -1

    def __summation(self, inputs, weights): ##TODO: Summation
        #inputs shape = (inputSize,1) weights = (inputSize,hiddenLayerSize)
        # sigma x_i*w_ij-theta_i
        sum = 0
        for i in range(len(inputs)):
            for j in range(len(weights)):
                np.sum((sum,(inputs[i]*weights[i][j])-self.thresholds[i]))
        return sum
    def __outputErr(self, desired, actual):
        return (desired-actual)
    def __errorGradient(self, output, outputErr, hidden, nextLayerOutput=0, nextLayerOutputErr=0): ##TODO: Error Gradient
        if(hidden):
            return (output-(1-output)*self.__summation(self.__errorGradient(nextLayerOutput,nextLayerOutputErr,False),self.weights))
        else:
            return (output-(1-output)*outputErr)
    def __deltaWeight(self, output, errorGradient):
        return (self.learningRate*output*errorGradient)
    def __genDeltaWeight(self, prevWeight, output, errorGradient):
        return ((self.momentum*prevWeight)+(self.learningRate*output*errorGradient)) #RH side is the original delta formula
    def __sumSquaredErrors(self, desired, actual):
        (1/self.outputLayer)*self.__summation((desired-actual)**2) ##TODO: Sum of Squared Errors
        pass

    def __printOut(self):
        pass

    def train(self, accelerated):
        x = 0
        while x <= self.epochs:
            print("Epoch: ", x)
            for i in self.weights: #Layer 
                print("Layer: ", i)
                for j in i: #Node
                    print("Node: ", j)
                    self.activateNode("sigmoid", self.inputs, j)
            print("----------")
            #TODO: Calculate Error Gradient
            #TODO: Accelerated/Non-accelerated Weight Correction
            if(accelerated):
                #Generalized Delta Rule
                pass
            else:
                #Regular Delta Rule
                pass
            #TODO: Calculate Sum of Squared Errors
            x+=1
    def test(self):
        pass

from operator import index
from typing import Type
from nnet import Perceptron, rand

inputs=[]
outputLayerSize = rand.randint(1,5)
hiddenLayers = []
hiddenLayers.append(rand.randint(1,5))
numInputs = rand.randint(1,5)
for i in range(numInputs):
    inputs.append(rand.randint(0,10))
for i in range(hiddenLayers[0]):
    hiddenLayers.append(rand.randint(1,10))

print(inputs) #crisp inputs
print("----------")
print(hiddenLayers) #num nodes in each
print("----------")
print(outputLayerSize) #num nodes in output
print("----------") 

net = Perceptron(inputs, hiddenLayers, outputLayerSize)
for i in range(len(net.weights)-1):
    print("Layer: ", i+1)
    for j in range(len(net.weights[i])):
        print(" Node: ", j+1)
        for k in range(len(net.weights[i][j])):
            try: 
                print("     Weight: ", k+1)
                raise TypeError
            except TypeError:
                print("         ",net.weights[i][j][k])
print()
for i in range(len(net.thresholds)):
    print("Layer: ", i+1)
    for j in range(len(net.thresholds[i])):
        print(" Node/Threshold: ", j+1)
        print("     ",net.thresholds[i][j])
print()
print("Layer: ", len(net.weights))
for j in (net.weights[-1]):
    print(" Node/Weight: ")
    print("     ",j)
#net.__summation(inputs, net.weights)
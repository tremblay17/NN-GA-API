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
print(net.weights)
print(net.thresholds)
print(net.hiddenLayers)
print(net.outputLayer)
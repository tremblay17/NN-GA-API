from nnet import NeuralNet, rand

inputs=[]
outputLayerSize = rand.randint(1,6)
hiddenLayers = []
hiddenLayers.append(rand.randint(1,5))
numInputs = rand.randint(6,15)
for i in range(numInputs):
    inputs.append(rand.randint(0,10))
for i in range(hiddenLayers[0]):
    hiddenLayers.append(rand.randint(1,10))

print(inputs)
print("----------")
print(hiddenLayers)
print("----------")
print(outputLayerSize)
print("----------")

net = NeuralNet(inputs, hiddenLayers, outputLayerSize)
net.train()
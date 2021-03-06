import Layer
import OutputLayer
import numpy as np
import copy

class Network:
    def __init__(self, inputSize, outputSize, hiddenLayerSizes, dropMode = None, activationFunction="Sigmoid"):
        self.Layers = []

        layerSizes = [inputSize]
        layerSizes.extend(hiddenLayerSizes)
        layerSizes.append(outputSize)

        for i in range(0, len(layerSizes) - 1):
            dropoutRate = .5 if dropMode == 'Dropout' else 0
            dropconnectRate = .5 if dropMode == 'Dropconnect' else 0
            newLayer = Layer.Layer(layerSizes[i], layerSizes[i+1], dropoutRate, activationFunction, dropconnectRate)
            newLayer.RandomizeWeights()
            self.Layers.append(newLayer)
        self.Layers[0].Mode = "Linear"
        self.Layers[0].DropoutRate = .2 if dropMode == 'Dropout' else 0

        self.OutputLayer = OutputLayer.OutputLayer(outputSize)

    def Feedforwad(self, input):
        nextLayerInput = input.copy()
        for layer in self.Layers:
            nextLayerInput = layer.Feedforward(nextLayerInput)
        self.OutputLayer.Activate(nextLayerInput)
        return self.OutputLayer.GetOutput()

    def GetError(self, expected):
        return self.OutputLayer.GetError(expected)

    def Backprop(self, expected, learningRate):
        gradients = self.GetGradients(expected)
        self.ApplyGradients(gradients, learningRate)

    def GetGradients(self, expected):
        gradients = []
        nextLayerDeltas = self.OutputLayer.GetDeltas(expected)
        for layer in reversed(self.Layers):
            gradient = np.dot(nextLayerDeltas.reshape(nextLayerDeltas.size, 1), layer.Activations.reshape(1, layer.Activations.size))
            #Set the gradient of any weights that were removed for Dropconnect to 0
            if(layer.DropconnectRate > 0):
                gradient = np.multiply(gradient, layer.DropconnectMask)
            #Insert the gradient at the beginning of the list, so that it ends up in the correct order
            gradients.insert(0, gradient)
            nextLayerDeltas = layer.MakeDeltas(nextLayerDeltas)
        return gradients

    def ApplyGradients(self, gradients, learningRate):
        for i in range(len(gradients)):
            layer = self.Layers[i]
            gradient = gradients[i]
            layer.Weights = np.add(layer.Weights, np.multiply(gradient, learningRate))

    def RemoveDrop(self):
        for layer in self.Layers:
            layer.Weights = np.multiply(layer.Weights, (1 - layer.DropoutRate))
            layer.DropoutRate = 0

            layer.Weights = np.multiply(layer.Weights, (1 - layer.DropconnectRate))
            layer.DropconnectRate = 0

    def MakeCopy(self):
        return copy.deepcopy(self)
import Network
import numpy as np
import Settings
import random

class RunManager:
    def __init__(self, trainingSet, validationSet, dropMode, fileManager):
        self.trainingSet = trainingSet
        self.validationSet = validationSet

        self.net = Network.Network(len(trainingSet[0][0]), len(trainingSet[0][1]), [400, 400], dropMode)
        self.epoch = 0
        self.fileManager = fileManager

        self.bestAccuracy = 0
        self.bestEpoch = 0
        self.bestNetwork = None

    def Train(self):
        minibatchGradients = None
        count = 0
        stillImproving = True

        while(stillImproving):
            random.shuffle(self.trainingSet)
            correct = 0

            for image in self.trainingSet:
                #Run the network on the image
                input = np.array(image[0])
                expected = np.array(image[1])
                output = self.net.Feedforwad(input)

                category = np.argmax(expected)
                classify = np.argmax(output)
                if(category == classify):
                    correct += 1
                count += 1

                #Get the gradients from this image
                gradients = self.net.GetGradients(expected)
                if(minibatchGradients is None):
                    minibatchGradients = gradients
                else:
                    for i in range(len(gradients)):
                        minibatchGradients[i] = np.add(minibatchGradients[i], gradients[i])

                #If we've reached the end of a minibatch, apply the gradients to the network
                if(count % Settings.MinibatchSize == 0):
                    for i in range(len(minibatchGradients)):
                        #Find the average gradients
                        minibatchGradients[i] = np.multiply(minibatchGradients[i], 1 / Settings.MinibatchSize)
                    #Apply the gradients to the network
                    self.net.ApplyGradients(minibatchGradients, Settings.LearningRate)
                    #Reset the gradients
                    minibatchGradients = None

            #Report the performance of the network at the end of each epoch
            trainingAccuracy = correct / len(self.trainingSet)
            validAccuracy = self.GetAccuracy(self.validationSet)
            performanceString = str(self.epoch) + ',' + str(trainingAccuracy) + ',' + str(validAccuracy)
            print(performanceString)
            self.fileManager.Write(performanceString)

            stillImproving = self.CheckImprovement(validAccuracy)
            self.epoch += 1

        #Once we're done training, reset to the best performing network
        self.net = self.bestNetwork

    def GetAccuracy(self, set):
        network = self.net.MakeCopy()
        network.RemoveDrop()
        correctCount = 0

        for image in set:
            input = np.array(image[0])
            expected = np.array(image[1])
            output = network.Feedforwad(input)

            if np.argmax(expected) == np.argmax(output):
                correctCount += 1

        return correctCount / len(set)

    def CheckImprovement(self, validAccuracy):
        stillImproving = True
        if(validAccuracy > self.bestAccuracy):
            self.bestAccuracy = validAccuracy
            self.bestEpoch = self.epoch
            self.bestNetwork = self.net.MakeCopy()
        elif(self.epoch - self.bestEpoch >= Settings.EarlyStopPatience):
            stillImproving = False
        return stillImproving
import GetInput
import Network
import numpy as np
import random

def GetAccuracy(set, network):
    correctCount = 0
    dropoutRate = network.Layers[0].DropoutRate
    dropoutRate2 = network.Layers[1].DropoutRate

    network.Layers[0].DropoutRate = 0
    network.Layers[1].DropoutRate = 0

    for image in set:
        input = np.array(image[0])
        expected = np.array(image[1])
        output = network.Feedforwad(input)

        if np.argmax(expected) == np.argmax(output):
            correctCount += 1

    network.Layers[0].DropoutRate = dropoutRate
    network.Layers[1].DropoutRate = dropoutRate2

    return correctCount / len(set)


def Run(trainingSet, validationSet, testingSet):


data = GetInput.LoadAllCategories("D:/Chris/Dropbox/Dropbox/School/ANN/SmallSet", "Full")
random.shuffle(data)
data = data[:]
print(len(data))
print(len(data[0][0]))

firstValidIndex = int(len(data) * .7)
firstTestIndex = int(len(data) * .85)

trainingSet = data[:firstValidIndex]
validationSet = data[firstValidIndex:firstTestIndex]
testingSet = data[firstTestIndex:]

numRuns = 10
accuracySum = 0
maxSum = 0
useDropout = True

for run in range(numRuns):
    net = Network.Network(len(data[0][0]), len(data[0][1]), [400,400], useDropout)

    groupCount = 0
    groupSize = 500

    count = 0
    correct = 0

    runAcc = 0
    maxAcc = 0

    for i in range(100):
        for image in trainingSet:
            input = np.array(image[0])
            expected = np.array(image[1])
            output = net.Feedforwad(input)
            error = net.GetError(expected)
            net.Backprop(expected, .01)

            category = np.argmax(expected)
            classify = np.argmax(output)

            if category == classify:
                correct += 1
            count += 1

            if count == groupSize:
                print(groupCount, end= "\t")
                print(correct / count)

                groupCount += 1
                correct = 0
                count = 0

        validAccuracy = GetAccuracy(validationSet, net)
        if validAccuracy > maxAcc:
            maxAcc = validAccuracy

    accuracySum += validAccuracy
    maxSum += maxAcc

print(accuracySum / numRuns)
print(maxSum / numRuns)
print(useDropout)


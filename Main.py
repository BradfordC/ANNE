import GetInput
from RunManager import RunManager
import random

data = GetInput.LoadAllCategories("D:/Chris/Dropbox/Dropbox/School/ANN/SmallSet", "Full")
random.shuffle(data)
print(len(data))
print(len(data[0][0]))

firstValidIndex = int(len(data) * .7)
firstTestIndex = int(len(data) * .85)

trainingSet = data[:firstValidIndex]
validationSet = data[firstValidIndex:firstTestIndex]
testingSet = data[firstTestIndex:]

numRuns = 2
accuracySum = 0
useDropout = True

for i in range(numRuns):
    manager = RunManager(trainingSet, validationSet, useDropout)
    manager.Train()
    testAccuracy = manager.GetAccuracy(testingSet)
    accuracySum += testAccuracy
    print(testAccuracy)
    print()

print(accuracySum / numRuns)
print(useDropout)


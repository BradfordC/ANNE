import GetInput
import Settings
from RunManager import RunManager
from FileManager import FileManager
import random

data = GetInput.LoadAllCategories("D:/Chris/Dropbox/Dropbox/School/ANN/SmallSet", "Full")
random.shuffle(data)
print(len(data))
print(len(data[0][0]))
print(Settings.Mode)

firstValidIndex = int(len(data) * .7)
firstTestIndex = int(len(data) * .85)

trainingSet = data[:firstValidIndex]
validationSet = data[firstValidIndex:firstTestIndex]
testingSet = data[firstTestIndex:]

accuracySum = 0
useDropout = True if Settings.Mode == 'Dropout' else False

for i in range(Settings.Runs):
    print('Run', i)
    fileManager = FileManager('D:/Chris/Dropbox/Dropbox/School/ANN/Results/' + Settings.Mode + str(i) + '.csv')

    manager = RunManager(trainingSet, validationSet, useDropout, fileManager)
    manager.Train()
    testAccuracy = manager.GetAccuracy(testingSet)
    fileManager.Write(testAccuracy)

    accuracySum += testAccuracy
    print(testAccuracy)
    print()

print(accuracySum / Settings.Runs)
print(useDropout)


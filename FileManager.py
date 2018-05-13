import os

class FileManager:
    def __init__(self, filePath):
        self.filePath = filePath

        #Delete the file if it exists
        if(os.path.exists(filePath)):
            os.remove(filePath)

        #Create the directory for the file if it doesn't exist
        dir = os.path.split(filePath)[0]
        if(not os.path.exists(dir)):
            os.makedirs(dir)

    def Write(self, string):
        string = str(string)
        with open(self.filePath, 'w') as file:
            file.write(string + '\n')
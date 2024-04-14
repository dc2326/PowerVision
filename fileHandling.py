from datetime import datetime

class NetListFile:

    def __init__(self, fileName = datetime.now().strftime("%d%m%Y_%H%M%S")+".net"):
        self.netName = fileName
        try:
            self.file = open(self.netName,"xt")
            self.file.close()
        except FileExistsError:
            print("This file exists already, please choose a different name.")
            exit()

    def write(self, contents):
        self.file = open(self.netName, 'wt')
        print(contents, file=self.file)
        self.file.close()

    def name(self):
        return self.netName
    




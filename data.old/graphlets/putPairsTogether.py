fileToWrite = open("allTogether.csv", 'w')
for i in range(0, 73):
    fileName = "orbital%02d.csv,allpairs,cv=0.1,B=n^0.6,Results.csv"%i
    fileToRead = open(fileName, 'r')
    fileToRead.readline()
    for line in fileToRead:
        fileToWrite.write("%d, %s"%(i, line))
    fileToRead.close()
fileToWrite.close()

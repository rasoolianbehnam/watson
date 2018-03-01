import numpy as np

chromosome = 14
cells = {"mit":[], "all":[], "rl":[], "call4":[]}
files = []

for cell in cells.keys():
    fileName = "chr%02d_chr%02d_%s.graphlet"%(chromosome, chromosome, cell)
    print fileName
    file = open(fileName, 'r')
    file.readline()
    for line in file:
        numbers = [ int(x) for x in line.split(';')]
        #print len(numbers)
        cells[cell].append(numbers)

for cell in cells.keys():
    cells[cell] = np.array(cells[cell])
    print cells[cell]

for i in range(0, 73):
    file = open("orbital%02d.csv"%i, 'w')
    file.write("mit, all, rl, call4\n")
    N = 100000
    for cell in cells.keys():
       N = np.min([cells[cell].shape[0], N]) 
    for j in range(N):
        toWrite = "%d, %d, %d, %d\n"% (cells['mit'][j, i],
                cells['all'][j, i],
                cells['rl'][j, i],
                cells['call4'][j, i]
                )
        if j == 10:
            print toWrite
        file.write(toWrite)
    file.close()

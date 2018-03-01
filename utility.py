import csv
import numpy as np
import os, errno
import cv2
import sys
from scipy.stats.stats import pearsonr   
import matplotlib.pyplot as plt
#TODO: not used
def eigendecompose(a):
    w, v = np.linalg.eig(a)
    max = np.abs(np.max(w))
    min = np.abs(np.min(w))
    t = np.max([min, max])
    w = w / t
    v = v * t * t
    print "result of multiplying eigenvalues and eigenvectors:"
    print v.dot(np.diag(w)).dot(np.linalg.inv(v))*t
    return w, v/t/t

def writeMatToFile(mat, fileName, delimiter=','):
    """
    Function:
    ---------
    Takes a matrix and writes it as a text matrix
    where each row is separated by a new line
    and each two cells in a row are separated
    by the specified delimiter.

    Parameters:
    -----------
    mat : ndarray(m, n) 
        The input matrix

    fileName: string
        name of the file to write to

    delimiter: string
        the delimiter that separates the cells in
        each row.
    """
    file = open(fileName, 'w')
    n, m = mat.shape
    for i in range(n):
        for j in range(m):
            file.write("%f"%mat[i, j])
            if j < m-1:
                file.write(delimiter)
        file.write("\n")
    file.close()

def pyrProcess(original):
    original = cv2.pyrDown(original)
    original = cv2.pyrUp(original)
    return scale(original)

def scale(m):
    min = np.min(m) + 1e-5
    max = np.max(m)
    n = (1. * m - min) / (max - min) 
    return n

#used to be simple read
def readMat(fileName, delimiter="\t", ignoreHeader=False):
    print "file directory:", fileName
    if not os.path.isfile(fileName):
        print "File %s does not exist"%fileName
        sys.exit(2)
    file = open(fileName, 'r')
    if ignoreHeader:
        file.readline()
    x = []
    for line in file:
        strings = line.split(delimiter)
        lineNums = []
        for i in range(0, len(strings)-1):
            num = strings[i]
            lineNums.append(float(num))
        x.append(lineNums)
    observed = np.array(x).astype("float")
    return observed

def readYaml(filename, res, delimiter="\t"):
    file = open(filename, 'r')
    n = 0
    lines = []
    for line in file:
        splitted = line.split(delimiter)
        #print splitted
        i = int(splitted[0]) / res
        j = int(splitted[1]) / res
        freq = float(splitted[2])
        n = np.max([i, j, n])
        #print n
        lines.append((i, j, freq))
    img = np.zeros((n+1, n+1))
    #print img.shape
    for line in lines:
        #print line
        img[line[0], line[1]] = line[2]
        img[line[1], line[0]] = line[2]
    img  = removeBlankRowsAndColumns(img)
    return img

def removeBlankRowsAndColumns(mat, thresh=5):
    n, m = mat.shape
    symmetric = True
    if n != m:
        print("Matrix is not symmetric!")
        symmetric = False
    matrixWithRowAndColumnsRemoved = mat
    blankRows = []
    for i in range(0, n):
        if np.sum(mat[i, :] > 0) <= thresh:
            blankRows.append(i)
    blankRows = sorted(blankRows, reverse=True)
    print "blonk rows:", blankRows
    if symmetric:
        for i in blankRows:
            matrixWithRowAndColumnsRemoved = np.delete(matrixWithRowAndColumnsRemoved, i, 0)
            matrixWithRowAndColumnsRemoved = np.delete(matrixWithRowAndColumnsRemoved, i, 1)
    else:
        blankCols = []
        for i in range(0, m):
            if np.sum(mat[:, i] > 0) <= thresh:
                blankCols.append(i)
        blankCols = sorted(blankCols, reverse=True)
        print "blonk cols:", blankCols
        for i in blankRows:
            matrixWithRowAndColumnsRemoved = np.delete(matrixWithRowAndColumnsRemoved, i, 0)
        for i in blankCols:
            matrixWithRowAndColumnsRemoved = np.delete(matrixWithRowAndColumnsRemoved, i, 1)
    print "size of new matrix:", matrixWithRowAndColumnsRemoved.shape
    return matrixWithRowAndColumnsRemoved

def pearson(counter, iterations=1):
    n = counter.shape[0]
    counter2 = np.zeros((n, n))
    for i in range(0, n):
        for j in range(i, n):
            counter2[i, j] = counter2[j, i] = pearsonr(counter[i, :], counter[:, j])[0]
    counter = counter2
    return counter

def setDiagonals(mat, depth, value):
    n, m  = mat.shape
    if n != m:
        print("matrix is not symmentric. Exitting.")
        return
    for i in range(depth):
        indices = np.arange(n-i)
        mat[indices, indices+i] = value
        mat[indices+i, indices] = value


def scn(a, minDiff=1e-3, numIterations=1000):
    a = a * 1.
    n = a.shape[0]
    counter = 0
    while True and counter < numIterations:
        #print("At iteration %d"%counter)
        if counter % 2:
            sumCols = np.sqrt(np.sum(a*a, axis=0))
            #print "sum of colums: ", sumCols
            for i in range(0, n):
                #print "a[:, i] = ", a[:, i]
                #print "sum cols = ", sumCols[i]
                #print "division = ", a[:, i] / sumCols[i]
                a[:, i] = a[:, i]*1. / sumCols[i]
        else:
        #print "a = ", a
            sumRows = np.sqrt(np.sum(a*a, axis=1))
            #print "sum of rows: ", sumRows
            for i in range(0, n):
                a[i, :] = a[i, :] / sumRows[i]
        diff = np.max(np.abs(a.T - a))
        #print("difference between matrices: %f"%diff)
        if diff < minDiff:
            break
        counter += 1
    #print "a = ", a
    for i in range(n):
        a[i, :] = a[:, i]
    return a

def normalize(heatmap):
    return (heatmap - np.mean(heatmap) ) / np.std(heatmap)
def showImages(imageList, rows):
    cols = (len(imageList) + rows - 1) / rows
    print "Number of columnts:", cols
    count = 1
    for image in imageList:
        ax = plt.subplot(rows, cols, count)
        count += 1
        ax.imshow(image)
    plt.show()
def cvShowImages(imageList):
    count = 1
    for image in imageList:
        cv2.imshow("image %d"%count, image)
        count += 1
    cv2.waitKey()

def convertBinaryMatToOrcaReadable(image, outputFileName=None):
    """
    function: prints the indices i+1, j+1 of the
    positive cells in the image image[i, j] > 0
    to a file or standard output.
    the indices start from 1.

    Parameters
    ----------
    image: should be SYMMETRIC

    outputFileName

    Returns:
    --------
    None
    """
    positives = np.where(image)
    if outputFileName != None:
        file = open(outputFileName, 'w')
    for index in range(len(positives[0])):
         i = positives[0][index]
         j = positives[1][index]
         if i < j:
             if outputFileName != None:
                 file.write("%d, %d\n"%(i+1, j+1))
             else:
                 print i+1, j+1


def writeYamlToMat(fileName, symmetric=False):
    print "file name =", fileName
    file = open(fileName, 'r')
    lines = []
    min_rows = 1000000000
    min_cols = 1000000000
    max_rows = -1
    max_cols = -1
    for line in file:
        #print line
        try:
            spl = line.split("\t")
            spl = [int(spl[0]), int(spl[1])]
            #print "spl = ", spl
            lines.append(spl)
            if (spl[0] < min_rows):
                min_rows = spl[0]
            if (spl[1] < min_cols):
                min_cols = spl[1]
            if (spl[0] > max_rows):
                max_rows = spl[0]
            if (spl[1] > max_cols):
                max_cols = spl[1]
            #print spl
        except:
            continue
    print "min_rows, max_rows:"
    print min_rows, max_rows
    print "min_cols, max_cols:"
    print min_cols, max_cols
    
    if max_rows < 0 or max_cols < 0:
        print("file seems not to be in the correct format. exitting function.")
        return 
    if symmetric:
        all_max = np.max([max_rows, max_cols]) + 1
        counter = np.zeros((all_max, all_max), dtype=np.float32)
    else:
        counter = np.zeros((max_rows+1, max_cols+1), dtype=np.float32)

    for line in lines:
        counter[line[0], line[1]] += 1
        if symmetric:
            counter[line[1], line[0]] += 1
    print "size of original matrix:", counter.shape
    n = counter.shape[0]
    counter = removeBlankRowsAndColumns(counter)
    outputFileName =  fileName.replace('.yaml', '.mat')
    writeMatToFile(counter, outputFileName, delimiter='\t')
    #cv2.imshow("counter", scale(counter)*100)
    #cv2.waitKey()
def approveToContinue(message):
    print(message),
    answer = sys.stdin.readline().strip().lower()
    if answer != 'y' and answer != 'yes':
        print("Exitting."),
        sys.exit(2)
    return answer

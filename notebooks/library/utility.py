import csv
import numpy as np
import os, errno
#import cv2
import sys
import scipy.misc
from scipy.stats.stats import pearsonr   
import matplotlib.pyplot as plt
from matplotlib import colors
#from sets import Set

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

#TODO: not used
def eigendecompose(a):
    w, v = np.linalg.eig(a)
    max = np.abs(np.max(w))
    min = np.abs(np.min(w))
    t = np.max([min, max])
    w = w / t
    v = v * t * t
    print("result of multiplying eigenvalues and eigenvectors:")
    print(v.dot(np.diag(w)).dot(np.linalg.inv(v))*t)
    return w, v/t/t

def writeMatToImage(mat, fileName):
    scipy.misc.imsave(fileName, mat)
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

    Output:
        Nothing
    """
    file = open(fileName, 'w')
    print("writing to %s"%fileName)
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

def scale(m, epsilon=1e-5):
    """
    Function:
    ---------
    Takes a matrix and scales int so that all values
    fall between epsilon and 1.

    Parameters:
    -----------
    mat : ndarray(m, n) 
        The input ndarray

    epsilon: 
        The lower bound for values. In some applicatons,
        values of 0 will result in numerical problems so
        epsilon is defaulted to zero to prevent such
        problems.

    delimiter: string
        the delimiter that separates the cells in
        each row.

    Output:
        n: The thresholded numpy ndarray
    """
    min = np.min(m) + epsilon
    max = np.max(m)
    n = (1. * m - min) / (max - min) 
    return n

#used to be simple read
def readMat(fileName, delimiter="\t", ignoreHeader=False, remove_blanks=False, verbose=False):
    """
    Function:
    --------- 
    Reads a file and parses it into a numpy ndarray

    Parameters:
    -----------
    fileName

    delimiter

    ignoreHeader:
        ignore the first line of the file. Used when the
        columns have titles.

    remove_blanks:
        if True, calls 'removeBlankRowsAndColumns' to the
        matrix so as to remove rows and column that 
        cosist of all zero values.

    Output:
    ---------
    observed

    """
    if verbose:
        print("file directory:", fileName)
    if not os.path.isfile(fileName):
        print("File %s does not exist"%fileName)
        sys.exit(2)
    file = open(fileName, 'r')
    if ignoreHeader:
        file.readline()
    x = []
    for line in file:
        strings = line.split(delimiter)
        lineNums = []
        for i in range(0, len(strings)):
            num = strings[i]
            lineNums.append(float(num))
        x.append(lineNums)
    observed = np.array(x).astype("float")
    if remove_blanks:
        observed, a, b = removeBlankRowsAndColumns(observed)
    return observed

def readYamlIntensities(filename, res, delimiter="\t", remove_blanks=False, symmetric=False):
    """
    Function:
    --------- 

    Parameters:
    -----------

    Output:
    -----------

    """
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
        if symmetric:
            img[line[1], line[0]] = line[2]
    if remove_blanks:
        img, a, b  = removeBlankRowsAndColumns(img)
    return img
def getBlankRowsAndColumns(mat, thresh=1):
    n, m = mat.shape
    blankRows = Set([])
    blankCols = Set([])
    for i in range(0, n):
        if np.sum(mat[i, :] > 0) <= thresh:
            blankRows.add(i)
    for i in range(0, m):
        if np.sum(mat[:, i] > 0) <= thresh:
            blankCols.add(i)
    return blankRows, blankCols

def removeRowsAndColumns(mat, blankRows, blankCols):
    print("size of old matrix:", mat.shape)
    blankRows = sorted(blankRows, reverse=True)
    blankCols = sorted(blankCols, reverse=True)
    matrixWithRowAndColumnsRemoved = mat
    for i in blankRows:
        matrixWithRowAndColumnsRemoved = np.delete(matrixWithRowAndColumnsRemoved, i, 0)
    for i in blankCols:
        matrixWithRowAndColumnsRemoved = np.delete(matrixWithRowAndColumnsRemoved, i, 1)
    print("size of new matrix:", matrixWithRowAndColumnsRemoved.shape)
    return matrixWithRowAndColumnsRemoved

def removeBlankRowsAndColumns(mat, thresh=1):
    print("size of old matrix:", mat.shape)
    n, m = mat.shape
    blankRows, blankCols = getBlankRowsAndColumns(mat, thresh=thresh)
    symmetric = True
    if n != m:
        print("Matrix is not symmetric!")
        symmetric = False
    if symmetric:
        blankRows.update(blankCols)
        blankCols = blankRows
    matrixWithRowAndColumnsRemoved = removeRowsAndColumns(mat, blankRows, blankCols)
    print("size of new matrix:", matrixWithRowAndColumnsRemoved.shape)
    return matrixWithRowAndColumnsRemoved, Set(blankRows), Set(blankCols)

def pearson_nonsym(a, b):
    na, ma = a.shape
    nb, mb  = b.shape
    if (ma != mb):
        print("""Number of columns (%d and %d) not the same.
            picking the minimum value..."""%(ma, mb))
        m = np.min([ma, mb])
        a = a[:,:m]
        b = b[:,:m]
    out = np.zeros((na, nb))
    for i in range(0, na):
        for j in range(0, nb):
            out[i, j]  = pearsonr(a[i, :], b[j, :])[0]
    return out

def pearson_sym(counter):
    n = counter.shape[0]
    counter2 = np.zeros((n, n))
    for i in range(0, n):
        for j in range(i, n):
            counter2[i, j] = counter2[j, i] = pearsonr(counter[i, :], counter[:, j])[0]
    counter = counter2
    return counter

def pearson(a, b=None):
    if isinstance(b, np.ndarray):
        return pearson_nonsym(a, b)
    else:
        return pearson_sym(a)

def row_wise_pearson(a, b, check_for_zeros=True):
    n1, m1 = a.shape
    n2, m2 = b.shape
    n = np.min([n1, n2])
    m = np.min([m1, m2])
    a = a[:n, :m]
    b = b[:n, :m]
    counter2 = np.zeros(n)
    for i in range(n):
        if check_for_zeros:
            if np.sum(a[i, :]) == 0:
                continue
            if np.sum(b[i, :]) == 0:
                continue
        counter2[i] = pearsonr(a[i, :], b[i, :])[0]
    return counter2

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

def gaussian_normalize(heatmap):
    return (heatmap - np.mean(heatmap) ) / np.std(heatmap)

def _showImages(imageList, rows=None, cols=None, color_bar=False, titles=None, ax_labels=None, cmap=None, save_dir=None):
    if rows == None and cols == None:
        rows = 1
        cols = (len(imageList) + rows - 1) // rows
    elif cols == None:
        cols = (len(imageList) + rows - 1) // rows
    elif rows == None:
        rows = (len(imageList) + cols - 1) // cols
    print("Number of rows and columns: %d, %d"%(rows, cols))
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=.6)
    count = 0
    for image in imageList:
        if rows * cols == 1:
            ax = axes
        elif rows == 1 or cols == 1:
            ax = axes[count]
        else:
            ax = axes[count // cols, count % cols]
        m = ax.imshow(image, cmap=cmap, norm=colors.SymLogNorm(1))
        if titles != None:
            ax.title.set_text(titles[count])
        if ax_labels != None:
            ax.set(xlabel=ax_labels['x'][count], ylabel=ax_labels['y'][count])
        count += 1
    if color_bar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cb = fig.colorbar(m, cax=cax)
    if save_dir != None:
        plt.savefig(save_dir)
    plt.show()

def showImages(imageList, rows=None, cols=None, color_bar=False, titles=None, ax_labels=None, cmap="RdBu_r", save_dir=None):
    if isinstance(imageList, dict):
        images = []
        titles = []
        for key in imageList:
            images.append(imageList[key])
            titles.append(key)
        _showImages(images, rows=rows, cols=cols, \
                color_bar=color_bar, titles=titles, ax_labels=ax_labels, cmap=cmap \
                , save_dir=save_dir)
    else:
        _showImages(imageList, rows=rows, cols=cols, \
                color_bar=color_bar, titles=titles, ax_labels=ax_labels, cmap=cmap \
                , save_dir=save_dir)


def cvShowImages(imageList):
    count = 1
    for image in imageList:
        cv2.imshow("image %d"%count, image)
        count += 1
    cv2.waitKey()

def convertBinaryMatToOrcaReadable(image, outputFileName=None, x_beg=0, y_beg=0):
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
                 file.write("%d, %d\n"%(i+1+x_beg, j+1+y_beg))
             else:
                 print(i+1+x_beg, j+1+y_beg)


def readYaml(fileName, delimiter ="\t", symmetric=False, remove_blank=False):
    print("file name =", fileName)
    file = open(fileName, 'r')
    lines = []
    min_rows = 1000000000
    min_cols = 1000000000
    max_rows = -1
    max_cols = -1
    for line in file:
        #print line
        try:
            spl = [int(x) for x in line.split("\t")]
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
    print("min_rows = %d, max_rows = %d" %(min_rows, max_rows))
    print("min_cols = %d, max_cols = %d"%( min_cols, max_cols))
    if symmetric:
        print("Symmetric on")
    
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
    n = counter.shape[0]
    if remove_blank:
        counter, a, b = removeBlankRowsAndColumns(counter)
    return counter
    #outputFileName =  fileName.replace('.yaml', '.mat')
    #writeMatToFile(counter, outputFileName, delimiter='\t')
    #cv2.imshow("counter", scale(counter)*100)
    #cv2.waitKey()
def approveToContinue(message):
    print(message),
    answer = sys.stdin.readline().strip().lower()
    if answer != 'y' and answer != 'yes':
        print("Exitting."),
        sys.exit(2)
    return answer

def graphlet_distance(g1, g2):
    """
    ----------------------------------------------
    Function:                                    #
    ----------------------------------------------
    Receives two matrices of graphlets of size 
    N * O.  Each column correspons to the 
    orbital with the corresponsing index.

    ----------------------------------------------
    Inputs:                                      #
    ----------------------------------------------
    - g1 and g2: two N * O matrix as explained above

    ----------------------------------------------
    Returns:                                     #
    ----------------------------------------------
    - out: an N * 1 matrix where each cell 
    out[i, 0] is the graphlet signature
    DISSIMILARITY between the ith rows of g1 and g2.
    """
    raw_weights = np.array([2, 4, 3, 9, 6, 10, 12, 4,
                   24, 4, 10, 7, 6, 6, 28, 8,
                   14, 8, 5, 12, 9, 11, 32, 15,
                   12, 8, 22, 5, 9, 14, 11,
                   16, 18, 15, 45, 7, 9, 20,
                   13, 8, 20, 11, 15, 36, 15,
                   7, 9, 13, 22, 36, 28, 22,
                   10, 26, 36, 30, 8, 33, 15,
                   22, 26, 15, 12, 28, 26, 12,
                   26, 30, 56, 15, 28, 45, 50])
    raw_numbers = np.array([2, 2, 1, 3, 2, 2, 3, 1,
                   4, 1, 2, 1, 2, 2, 4, 2,
                   2, 1, 1, 2, 1, 1, 4, 1,
                   2, 1, 2, 1, 1, 2, 1,
                   2, 2, 1, 5, 1, 1, 2, 
                   1, 1, 2, 1, 1, 4, 1, 
                   1, 1, 1, 2, 3, 2, 2,
                   1, 2, 3, 2, 1, 3, 1,
                   2, 2, 1, 1, 2, 2, 1,
                   2, 2, 4, 1, 2, 3, 5])
    #print(raw_weights.shape)
    #print(raw_numbers.shape)
    try:
        n1, o1 = g1.shape
    except:
        o1 = g1.shape[0]
        g1 = g1.reshape(-1, o1)

    try:
        n2, o2 = g2.shape
    except:
        o2 = g2.shape[0]
        g2 = g2.reshape(-1, o2)

    n1, o1 = g1.shape
    n2, o2 = g2.shape
    n = np.min([n1, n2])
    o = np.min([o1, o2])
    g1 = g1[:n, :o]
    g2 = g2[:n, :o]

    normalized_weights = (raw_weights / raw_numbers)[:o].reshape(1, -1)
    normalized_weights = np.ones((n, 1)).dot(normalized_weights)
    normalized_weights = 1 - np.log(normalized_weights) / np.log(73)
    #print(normalized_weights.shape)
    pairwise_distance \
            = normalized_weights * np.abs(np.log(g1 + 1) - np.log(g2 + 1)) / \
            np.log(np.maximum(g1, g2) + 2)
    out = np.linalg.norm(pairwise_distance, axis=1) 
    return out

def full_graphlet_distance(g1, g2):
    """
    -------------------------------------------------------
    Function
    ------------------------------------------------------
    Calculate correlation between two matrices of graphlets
    -------------------------------------------------------
    Inputs:
    -------------------------------------------------------
    - g1, g2 : N * O matrices
    -------------------------------------------------------
    Returns:
    -------------------------------------------------------

    - out: an N * N matrix where each cell 
    out[i, j] is the graphlet DISSIMILARITY between
    orbit i in g1 and orbit j in g2.
    """
    n1, o1 = g1.shape
    n2, o2 = g2.shape
    assert(o1 == o2)
    out = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            out[i, j] = graphlet_distance(g1[i, :], g2[j, :])
    return out

def row_wise_graphlet_distance(g1, g2, check_for_zeros=True):
    """
    -------------------------------------------------------
    Function
    ------------------------------------------------------
    Calculate correlation between two matrices of graphlets
    -------------------------------------------------------
    Inputs:
    -------------------------------------------------------
    - g1, g2 : N * O matrices
    -------------------------------------------------------
    Returns:
    -------------------------------------------------------

    - out: an N * N matrix where each cell 
    out[i, j] is the graphlet DISSIMILARITY between
    orbit i in g1 and orbit j in g2.
    """
    n1, o1 = g1.shape
    n2, o2 = g2.shape
    n = np.min([n1, n2])
    o = np.min([o1, o2])
    out = np.zeros(n)
    for i in range(n):
        if check_for_zeros:
            if np.sum(g1[i, :]) == 0:
                continue
            if np.sum(g2[i, :]) == 0:
                continue
        out[i] = graphlet_distance(g1[i, :], g2[i, :])
    return out

def pair_orbits(graphlets, chromosome, delimiter=','):
    count = 0
    n = o = sys.maxint
    for key in graphlets:
        ni, oi = graphlets[key].shape
        n = np.min([n, ni])
        o = np.min([o, oi])
        print("shape of graphlet for chromosome %d of cell %s: \
                %d by %d"%(chromosome, key, n, o))
        count += 1
    print("number of matrices:", count)
    for orbit in range(0, 73):
        file_name = "data/chr%02d_orbit%02d.paired.csv"\
                %(chromosome, orbit)
        print(file_name)
        file = open(file_name, 'w')
        for key in graphlets:
            file.write(key)
            file.write(delimiter)
        file.write("\n")
        for i in range(n):
            for key in graphlets:
                file.write("%d"%graphlets[key][i, orbit])
                file.write(delimiter)
            file.write("\n")
        file.close()

def get_mic_from_file(file_name, delimiter=','):
    #print(file_name)
    file = open(file_name)
    line = [x for x in file.readline().split(delimiter)]
    #print(line)
    f1 = []
    f2 = []
    mic = []
    for line in file:
        content = [x for x in line.split(delimiter)]
        f1.append(content[0])
        f2.append(content[1])
        mic.append(float(content[2]))
    return f1, f2, mic

def log(image):
    return np.where(out > 0, np.log(out), 0)

def print_statistics(m1, text = "", print_results=True):
    max = np.nanmax(m1)
    min = np.nanmin(m1)
    mean = np.nanmean(m1)
    median = np.percentile(m1, 50)
    std = np.nanstd(m1)
    if (print_results):
        print("#-----------------------------------------------------------------------#")
        print("Stats for %s: "% text)
        print("max: %f | min: %f | median: %f | mean: %f | std: %f"%(max, min, median, mean, std))
        print("#-----------------------------------------------------------------------#")
    return max, min, mean, std

def t_test(m1, m2=None, one_sided=False):
    if m2 == None:
        diff = m1.reshape(1, -1)
    else:
        diff = (m1 - m2).reshape(1, -1)
    n = diff.shape[0] * diff.shape[1]
    #std = np.sqrt(np.sum(diff * diff) / (n - 1))
    std = np.std(diff, ddof=1)
    t0 = diff.mean() * np.sqrt(n) / std
    if one_sided:
        t0 = t0 / 2.
    return t0, scipy.stats.t.cdf(t0, n-1)

def is_larger(a, b, alpha=.01):
    if a.ndim == 1:
        m = 1
        n1 = len(a)
        n2 = len(b)
        n = np.min([n1, n2])
        a = a[:n]
        b = b[:n]
    else:
        m1, n1 = a.shape
        m2, n2 = b.shape

        m = np.min([m1, m2])
        n = np.min([n1, n2])
        
        a = a[:m, :n]
        b = b[:m, :n]

    d = a - b
    sd = np.std(d, ddof=1)
    avg = np.mean(d)
    t = avg / sd * np.sqrt(m*n)
    p_value = scipy.stats.t.cdf(t, n*m-1)
    if p_value < alpha:
        return -1
    p_value = scipy.stats.t.cdf(-t, n*m-1)
    if p_value < alpha:
        return 1
    return 0

def compare_pairwise(a, comparator=is_larger, labels=None):
    n = len(a)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            out[i, j] = comparator(a[i], a[j])
            out[j, i] = -comparator(a[i], a[j])
    if labels != None:
        ordered = [None for i in range(n)]
        sign = np.zeros(n, dtype='uint8')
        for i in range(n):
            position = np.sum(np.where(out > 0, 1, 0)[i])
            while ordered[position] != None:
                sign[position] = 1
                position += 1
            ordered[position] = labels[i]
        output_text = ""
        for i in range(n-1):
            letter = '<'
            if sign[i]:
                letter = '='
            output_text = "%s %10s %2s"%(output_text, ordered[i], letter)
        output_text = "%s %10s"%(output_text, ordered[n-1])
        out = (out, output_text)
    return out
            
def threshold_within_std(m1, k):
    min, max, mean, std = print_statistics(m1, print_results=False)
    return (m1 > mean - k * std) * (m1 < mean + k * std)

def kl_div(m1, m2, num_bins=10, same_range = False, normalization=False):
    ''''
    performs KL-divergence between two matrices m1 and m2.
    m1 and m2 should be vectors of size 1 * N.
    -----------------------------------------------------
    inputs:
    ----------------------------------------------------
    m1 , m2: matrices of size(1 * N)  
    '''
    n1 = m1.shape
    n2 = m2.shape
    N = np.min([n1, n2])
    m1 = m1[:N]
    m2 = m2[:N]
    if (normalization):
        print("normalizatoin is on")
        m1 = gaussian_normalize(m1)
        m2 = gaussian_normalize(m2)
    if same_range:
        max_val1 = np.max(m1)
        max_val2 = np.max(m2)
        max_val = np.max([max_val1, max_val2])
        min_val1 = np.min(m1)
        min_val2 = np.min(m2)
        min_val = np.min([min_val1, min_val2])
        bins = np.arange(0, 1, 1./num_bins) * (max_val - min_val) + min_val
        hist1 = np.histogram(m1, bins=bins)
        hist2 = np.histogram(m2, bins=bins)
    else:
        hist1 = np.histogram(m1, bins=num_bins)
        hist2 = np.histogram(m2, bins=num_bins)
    p = np.array(hist1[0], dtype='float32')
    q = np.array(hist2[0], dtype='float32')
    return np.sum(np.where((p*q) != 0,(p) * np.log10(p / q), 0))
def svd_reconstruct(mat, k, u='empty', sig='empty', v='empty'):
    if u == 'empty' or v == 'empty' or sig == 'empty':
        print("calculating u, sig and v ...")
        u, sig, v = np.linalg.svd(mat)
        print("finished with svd, commencing reconstruction ...")
    else:
        print("u, sig and v already provided, commencing reconstruction ...")
    #print(sig)
    if isinstance(k, int):
        iterable = range(k)
    else:
        iterable = k
    b = np.zeros_like(mat)
    for i in iterable:
        b += sig[i] * np.outer(u[:, i], v[i, :])
    return b, u, sig, v

def local_threshold(mat, k=1, method='max', t=1, \
                params=None, ignore_zeros=True, symmetric=False\
                , epsilon=1e-3):
    mat2 = np.zeros_like(mat)
    n, m = mat.shape
    if isinstance(k, tuple):
        if len(k) == 2:
            k11, k21 = k
            k12, k22 = k
        else:
            k11, k12, k21, k22 = k
    else:
        k11 = k12 = k21 = k22 = k
    print(k11, k12, k21, k22)
    for i in range(0, n):
        i_low = np.max([i - k11, 0])
        i_high = np.min([i + k12+1, n])
        j_beg = 0
        if symmetric:
            j_beg = i
        for j in range(j_beg, m-1):
            j_low = np.max([j - k21, j_beg])
            j_high = np.min([j + k22+1, m])
            temp = mat[i_low:i_high, j_low:j_high]
            #print(temp)
            if ignore_zeros:
                temp = temp[np.where(temp > 0)]
            if np.sum(temp) < epsilon:
                continue
            if method == 'max':
                condition = mat[i, j] == np.max(temp)
            elif method=='normal':
                condition = mat[i, j] > np.mean(temp) + t * np.std(temp)
            if condition:
                mat2[i, j] = condition
                if symmetric:
                    mat2[j, i] = condition
    if isinstance(params, tuple):
        mat3 = mat2 * 1.
        k = params[0]
        s = params[1]
        for i in range(0, n):
            i_low = np.max([i - k, 0])
            i_high = np.min([i + k + 1, n])
            for j in range(0, m):
                j_low = np.max([j - k+1, 0])
                j_high = np.min([j + k+1, m])
                if np.sum(mat3[i_low:i_high, j_low:j_high]) < s:
                    mat2[i, j] = 0
    return mat2

def _fromLowToHigh(i, l1, l2):
    k1 = 1
    while k1 < len(l1) and i > l1[k1, 0]:
                k1 += 1
    k1 -= 1
    if k1 > 1:
                remi = i % l1[k1, 0]
    else:
                remi = i
    
    if remi > l1[k1, 1]:
                print -1
    return l2[k1, 0] + remi

def _convertIndices(i, j):
        return (fromLowToHigh(i, nss_low, nss_high), fromLowToHigh(j, nss_low, nss_high))
def loadDataOld(halfSize=11, root="./", log_convert=False):
    print("Starting to load data...")
    map = {}
    nss_low = np.load("%s/length_low_res.npy"%root)
    nss_high = np.load("%s/length_high_res.npy"%root)
    mit_low = np.load("%s/mit_low_res.npy"%root)
    mit_high = np.load("%s/mit_high_res.npy"%root)   
    if log_convert:
        mit_low = np.log(mit_low + 1)
        mit_high = np.log(mit_high + 1)
    halfSize = 11
    n_low = mit_low.shape[0]
    n_high = mit_high.shape[0] 
    print(n_low, n_high)
    XXX = []
    YYY = []
    for chr1 in range(1, 24):
        for chr2 in range(chr1, 24):
            n1 = nss_low[chr1, 1]
            n2 = nss_high[chr1, 1]
            n = np.min([n1, n2])
            m1 = nss_low[chr2, 1]
            m2 = nss_high[chr2, 1]
            m = np.min([m1, m2])       
            beg_x_low = nss_low[chr1, 0]
            beg_y_low = nss_low[chr2, 0]
            beg_x_high = nss_high[chr1, 0]
            beg_y_high = nss_high[chr2, 0]       
            for i in range(n):
                for j in range(m):
                    xLow = beg_x_low+i
                    yLow = beg_y_low+j
                    xHigh = beg_x_high+i
                    yHigh = beg_y_high+j
                    if xLow < yLow:
                                                continue
                    #print(chr1, chr2, xLow, yLow, xHigh, yHigh)
                    X = mit_low[xLow-halfSize:xLow+halfSize+1, yLow-halfSize:yLow+halfSize+1]
                    Y = mit_high[xLow, yLow]
                    if X.shape[0] != 2*halfSize+1 or X.shape[1] != 2*halfSize+1:
                                                continue
                    map[(chr1, chr2, i, j)] = len(XXX)
                    XXX.append(X)
                    YYY.append(Y)
    print("Finished loading data.")
    return XXX, YYY, map
def extractContactMapFromLargeMatrix(mit_low, nss_low, chr1, chr2):
    begLowRow = nss_low[chr1, 0]
    begLowCol = nss_low[chr2, 0]
    n = nss_low[chr1, 1]
    m = nss_low[chr2, 1]
    return mit_low[begLowRow:begLowRow+n, begLowCol:begLowCol+m]

def cropToSameSize(mat1, mat2):
    n1, m1 = mat1.shape
    n2, m2 = mat2.shape
    n = np.min([n1, n2])
    m = np.min([m1, m2])
    return mat1[:n, :m], mat2[:n, :m]
    
def extractNeighborhood(origin, target, halfSize=11, cropToMin=True):
    n1, m1 = origin.shape
    n2, m2 = target.shape
    n = n1
    m = m1
    if cropToMin:
        print("Cropping to mininmum size ...")
        n = np.min([n1, n2])
        m = np.min([m1, m2])
        print("n = %d ; m = %d"%(n, m))
    else:
        assert n1 == n2
        assert m1 == m2
    Xout = []
    yOut = []
    pixelCache = []
    for i in range(halfSize, n-halfSize):
        for j in range(halfSize, m-halfSize):
            Xout.append(origin[i-halfSize:i+halfSize+1, j-halfSize:j+halfSize+1])
            yOut.append(target[i, j])
            pixelCache.append((i, j))
    numRows = n - 2 * halfSize
    numCols = m - 2 * halfSize
    #Xout = np.array(Xout).reshape(numRows, numCols, 2*halfSize+1, 2*halfSize+1, 1)
    #yOut = np.array(yOut).reshape(numRows, numCols)
    return Xout, yOut, pixelCache
            
def loadDataOld(halfSize=11, root="./", log_convert=False):
    print("Starting to load data...")
    nss_low = np.load("%s/length_low_res.npy"%root)
    nss_high = np.load("%s/length_high_res.npy"%root)
    mit_low = np.load("%s/mit_low_res.npy"%root)
    mit_high = np.load("%s/mit_high_res.npy"%root)   
    if log_convert:
        mit_low = np.log(mit_low + 1)
        mit_high = np.log(mit_high + 1)
    mit_low = (mit_low - mit_low.mean()) / mit_low.std()
    mit_high = (mit_high - mit_high.mean()) / mit_high.std()
    XX = []
    YY = []
    chrCache = []
    pixelCaches = []
    for chr1 in range(1, 24):
        for chr2 in range(chr1, chr1+1):
            lowConMap = extractContactMapFromLargeMatrix(mit_low, nss_low, chr1, chr2)
            highConMap = extractContactMapFromLargeMatrix(mit_high, nss_high, chr1, chr2)
            lowConMap, highConMap = cropToSameSize(lowConMap, highConMap) 
            Xout, yOut, pixelCache = extractNeighborhood(lowConMap, highConMap, halfSize=halfSize)
            #print(Xout.shape)
            #row, col, *rest = Xout.shape
            #map.append((row, col))
            #XXX.append(Xout.reshape(-1, 2*halfSize+1, 2*halfSize+1, 1))
            #YYY.append(yOut.reshape(-1))
            XX.append(Xout)
            YY.append(yOut)
            chrCache.append((chr1, chr2))
            pixelCaches.append(pixelCache)
    XXX = []
    YYY = []
    totalCache = []
    count = 0
    for i in range(len(XX)):
        for j in range(len(XX[i])):
            XXX.append(XX[i][j])
            YYY.append(YY[i][j])
            totalCache.append(chrCache[i] + pixelCaches[i][j])
    XXX = np.array(XXX).reshape(-1, 2*halfSize+1, 2*halfSize+1, 1)
    YYY = np.array(YYY)
    print("Finished loading data.")
    return XXX, YYY, (totalCache, nss_low, nss_high)




def reconstructFromPredictions(XXX, YYY, cache, chr1NotInclude=[], \
        chr2NotInclude=[], beg = 0, end=None):
    if end == None:
        end = len(XXX)
    reconstructed = {}
    cache, nss_low, nss_high = cache
    for i in range(beg, end):
        c1, c2, x, y = cache[i]
        if c1 in chr1NotInclude or c2 in chr2NotInclude:
            continue
        images = reconstructed.get((c1, c2), None)
        if images == None:
            nLow = nss_low[c1, 1]
            mLow = nss_low[c2, 1]
            nHigh = nss_high[c1, 1]
            mHigh = nss_high[c2, 1]
            n = np.min([nLow, nHigh])
            m = np.min([mLow, mHigh])        
            images = ((np.zeros((n, m)), np.zeros((n, m))))
        images[0][x, y] = XXX[i-beg, 11, 11, 0]
        images[1][x, y] = YYY[i-beg]
        reconstructed[(c1, c2)] = images
    return reconstructed

def findPixelNeighbor(px, py, nx, ny, image):
    halfSizeX = nx // 2
    xRemainder = nx % 2
    halfSizeY = ny // 2
    yRemainder = ny % 2
    w, h = image.shape

    extraXbefore = halfSizeX - px
    if extraXbefore < 0:
        extraXbefore = 0

    extraXafter = (px+halfSizeX+xRemainder) - (w-1)
    if extraXafter < 0:
        extraXafter = 0

    extraYbefore = halfSizeY - py
    if extraYbefore < 0:
        extraYbefore = 0

    extraYafter = (py+halfSizeY+yRemainder) - (h-1)
    if extraYafter < 0:
        extraYafter = 0
    
    #print(extraXbefore, extraXafter, extraYbefore, extraYafter)
    if extraYbefore + extraXbefore + extraYafter + extraXafter == 0:
        return image[px-halfSizeX:px+halfSizeX+xRemainder, py-halfSizeY:py+halfSizeY+yRemainder]
    else:
        #print("index out of bound")
        out = np.zeros((nx, ny))
        #print(px-halfSizeX+extraXbefore,px+halfSizeX+xRemainder-extraXafter, py-halfSizeY+extraYbefore,py+halfSizeY+yRemainder-extraYafter)
        cropped = image[px-halfSizeX+extraXbefore:px+halfSizeX+xRemainder-extraXafter, py-halfSizeY+extraYbefore:py+halfSizeY+yRemainder-extraYafter]
        #print(cropped.shape)
        out[extraXbefore:nx-extraXafter,extraYbefore:ny-extraYafter] = cropped
        return out

def cropNeighbors(image, nx, ny, sx, sy):
    halfSizeX = nx // 2
    xRemainder = nx % 2
    halfSizeY = ny // 2
    yRemainder = ny % 2
    w, h = image.shape
    print("expected: ", (w - nx) // sx + 1, (h - ny) // sy + 1)
    out = []
    c1 = 0
    for px in np.arange(halfSizeX, w-halfSizeX-xRemainder+1, sx):
        c1 += 1
        c2 = 0
        for py in np.arange(halfSizeY, h-halfSizeY-yRemainder+1, sy):
            c2 += 1
            out.append(findPixelNeighbor(px, py, nx, ny, image))

    cache = (w, h, nx, ny, sx, sy, px+halfSizeX+xRemainder, py+halfSizeY+yRemainder, c1, c2)
    print("actual:", c1, c2, len(out))
    return out, cache

def reconstruct(cropped, cache):
    w, h, nx, ny, sx, sy,lpx,lpy,c1,c2 = cache
    halfSizeX = nx // 2
    xRemainder = nx % 2
    halfSizeY = ny // 2
    yRemainder = ny % 2
    out = np.zeros((w, h))
    count = 0
    for px in np.arange(halfSizeX, w-halfSizeX-xRemainder, sx):
        for py in np.arange(halfSizeY, h-halfSizeY-yRemainder, sy):
            extraXbefore = halfSizeX - px
            if extraXbefore < 0:
                extraXbefore = 0

            extraXafter = (px+halfSizeX+xRemainder) - (w-1)
            if extraXafter < 0:
                extraXafter = 0

            extraYbefore = halfSizeY - py
            if extraYbefore < 0:
                extraYbefore = 0

            extraYafter = (py+halfSizeY+yRemainder) - (h-1)
            if extraYafter < 0:
                extraYafter = 0

            if extraYbefore + extraXbefore + extraYafter + extraXafter == 0:
                out[px-halfSizeX:px+halfSizeX+xRemainder, py-halfSizeY:py+halfSizeY+yRemainder] = cropped[count]
            else:
                #print("index out of bound")
                out = np.zeros((nx, ny))
                #print(px-halfSizeX+extraXbefore,px+halfSizeX+xRemainder-extraXafter, py-halfSizeY+extraYbefore,py+halfSizeY+yRemainder-extraYafter)
                out[px-halfSizeX+extraXbefore:px+halfSizeX+xRemainder-extraXafter, py-halfSizeY+extraYbefore:py+halfSizeY+yRemainder-extraYafter] = \
                    cropped[count][extraXbefore:nx-extraXafter,extraYbefore:ny-extraYafter]
            count += 1
    return out, cache


def loadData(nx, ny, sx, sy, root="./", log_convert=False):
    print("Starting to load data...")
    nss_low = np.load("%s/length_low_res.npy"%root)
    nss_high = np.load("%s/length_high_res.npy"%root)
    mit_low = np.load("%s/mit_low_res.npy"%root)
    mit_high = np.load("%s/mit_high_res.npy"%root)   
    if log_convert:
        mit_low = np.log(mit_low + 1)
        mit_high = np.log(mit_high + 1)
    mit_low = (mit_low - mit_low.mean()) / mit_low.std()
    mit_high = (mit_high - mit_high.mean()) / mit_high.std()
    out_low = []
    out_high = []
    cache = []
    for chr1 in range(1, 24):
        for chr2 in range(chr1, chr1+1):
            lowConMap = extractContactMapFromLargeMatrix(mit_low, nss_low, chr1, chr2)
            highConMap = extractContactMapFromLargeMatrix(mit_high, nss_high, chr1, chr2)
            lowConMap, highConMap = cropToSameSize(lowConMap, highConMap) 
            low_features, cache_low = cropNeighbors(lowConMap, nx, ny, sx, sy)
            high_features, cache_high = cropNeighbors(highConMap, nx, ny, sx, sy)
            cache.append((len(low_features), cache_low, cache_high))
            #print(len(low_features))
            #print(len(high_features))
            #print("****")
            out_low.extend(low_features)
            out_high.extend(high_features)
    out_low = np.array(out_low)
    out_high = np.array(out_high)
    print("finished loading data")
    return out_low, out_high, cache

def manova(data, alpha=.05, method='wilk'):
    D = -1
    M = len(data)
    for cell in data:
        assert D < 0 or data[cell].shape[1] == D
        D = data[cell].shape[1]
    averages = np.zeros((M, D))
    counts = np.zeros((M, 1))
    count = 0
    for cell in data:
        averages[count] = np.mean(data[cell], axis=0)
        counts[count, 0] = data[cell].shape[0]
        count += 1
    mu = np.mean(averages, axis=0, keepdims=True)
    T = np.zeros((D, D))
    for cell in data:
        diff = data[cell] - mu
        T += diff.T.dot(diff)
    #print("T:", T)
    diff = averages - mu
    H = diff.T.dot(diff * counts)
    #print("H:", H)
    E = T - H
    #print("E:", E)

    print("Method Used: %s"%method.upper())
    n = np.sum(counts)
    k = D
    m = len(data)
    print("n: ", n)
    print("k: ", k)
    print("m: ", m)
    vals = np.real(np.linalg.eigvals(H.dot(np.linalg.inv(E))))
    if method.lower() == 'wilk':
        detE = np.linalg.det(E)
        detEH = np.linalg.det(E+H)
        #print(tmp)
        #print(detE, detEH)
        gamma = np.prod(1 / (1+vals))
        a = n - m - (k-m+2.)/2.
        b_num = k**2*(m-1)**2-4.
        b_denom = k**2+(m-1)**2-5.
        if b_denom <= 0:
            b = 1
        else:
            b = np.sqrt(b_num / b_denom)
        c = (k * (m-1)-2) / 2.
        df1 = k*(m-1)
        df2 = a*b - c
        F = (gamma**(-1./b) - 1) * df2 / df1
        print("a: ", a)
        print("b_num", b_num)
        print("b_denom", b_denom)
        print("b: ", b)
        print("c: ", c)
        print("gamma: ", gamma)
    s = np.min([k, m-1])
    t = (np.abs(k-m+1)-1) / 2.
    u = (n-m-k-1)/2.
    print('s', s)
    print('t', t)
    print('u', u)
    if method.lower() == 'hotelling':
        df1 = s*(2*t+s+1)
        df2 = 2*(s*u+1)
        T20 = np.abs(np.trace(H.dot(np.linalg.inv(E))))
        T20 = np.sum(vals)
        F = (T20 * df2 * 1.) / (s * df1)
        print('T20: ', T20)
    if method.lower() == 'pillai':
        df1 = s * (2*t+s+1)
        df2 = s * (2*u+s+1)
        V = np.sum(vals / (1+vals))
        print('V: ', V)
        F = (V * df2) / ((s-V) * df1)
    print("df1", df1)
    print("df2", df2)
    print("F", F)
    print("alpha:", alpha)
    print("F-crit:", scipy.stats.f.ppf(1-alpha, df1, df2))
    print("p-value:", scipy.stats.f.sf(F, df1, df2))

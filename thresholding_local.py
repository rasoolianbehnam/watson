
# coding: utf-8

# # Local Thresholding
# In this notebook, I extract orbits for a signle chromosome in all 4 cell lines, using a
# different method of thresholding.
# In order to do this, I slide a kernel of size k 
# through each pixel and check if it has a 
# particular property. If the property is satisified, then the pixel is set to 1,
# otherwise it is set to 0. 
# The two properties that I experiment with are *maximum thresholding* and *normal 
# thresholding*.
# 1. **Maximum Thresholding**: in max thresholding, if the pixel is the maximum of its 
# neighbors with respect to the kernel, then it is set.
# 2. **Normal Thresholding**: in normal thresholding, if the pixel is larger that the 
# average of the neighbors then it is set.
# 
# first I perform the most lates normalization method that I have found to be most useful: a combination of gaussian normalization, SCN and pearson correlation, in the same order mentioned. The normalized results are of type ```pearson(scn(gaussian(graph)))```.
# 
# Then I threshold the results by setting pearson values larger than 0 to 1 and the rest to 0.
# In order to extract graphlets, I used the *orca* package in R. the script `./rscript.r`
# read a binary thresholded matrix representing an unweighted graph and outputs orbtis 0 
# through 72 for each node in the network.
# After thredholding the matrices, I save them so that orca in R can read it and extract orbits.
# 
# By repeating the procedure explained above for all 23 chromosomes, we will have graphlets
# for all contact maps.

# In[5]:


import numpy as np
import cv2
from utility import *
import matplotlib.pyplot as plt
from iced import normalization
from iced import filter
import os


# ## Initiating matrices

# In[18]:

def main():
    chr1 = 2
    for chr1 in range(1, 23):
        chr2 = chr1
        print "chromosome 1: %s, chromosome 2: %s"%(chr1, chr2)
        #print observed_mit_original.shape

        mit = readMat("../data/extracted_all_contact_MIT/chr_%02d_%02d_500kb.mat"%              (chr1, chr2), "\t")
        call4 = readMat("../data/extracted_all_contact_call4/chr_%02d_%02d_500kb.mat"%                (chr1, chr2), "\t")
        rl = readMat("../data/extracted_all_contact_RL/chr_%02d_%02d_500kb.mat"%(chr1, chr2), "\t")

        all = readMat("../data/extracted_all_contact_ALL/chr_%02d_%02d_500kb.mat"%(chr1, chr2), "\t")

        #There are some blank rows and columns in the matrix. let's remove them.
        blankRows0, blankCols0 = getBlankRowsAndColumns(mit)
        blankRows1, blankCols1 = getBlankRowsAndColumns(all)
        blankRows2, blankCols2 = getBlankRowsAndColumns(rl)
        blankRows3, blankCols3 = getBlankRowsAndColumns(call4)
        blankRows = Set([])
        blankCols = Set([])
        blankRows.update(blankRows0)
        blankRows.update(blankRows1)
        blankRows.update(blankRows2)
        blankRows.update(blankRows3)
        blankCols.update(blankCols0)
        blankCols.update(blankCols1)
        blankCols.update(blankCols2)
        blankCols.update(blankCols3)
        mit = removeRowsAndColumns(mit, blankRows, blankCols)
        all = removeRowsAndColumns(all, blankRows, blankCols)
        rl = removeRowsAndColumns(rl, blankRows, blankCols)
        call4 = removeRowsAndColumns(call4, blankRows, blankCols)


        mit = local_threshold(scn(mit), k = 2, method='normal', t = 1) 
        all = local_threshold(scn(all), k = 2, method='normal', t = 1)
        rl = local_threshold(scn(rl), k = 2, method='normal', t = 1) 
        call4 = local_threshold(scn(call4), k = 2, method='normal', t = 1) 


        images = [ 
                mit, all, rl, call4
                , (mit == all) * (mit != 0) * (all != 0) , (mit == rl) * (mit != 0) * (rl != 0) 
                , (mit == call4) * (mit != 0) * (call4 != 0) 
                , (rl == call4) * (rl != 0) * (call4 != 0) 
                , all != mit, rl != mit, call4 != mit, rl != call4
                 ] 
        titles = ['mit'
                , 'all'
                , 'rl'
                , 'call4'
                , 'mit == all'
                , 'mit == rl'
                , 'mit == call4'
                , 'rl == call4'
                , 'mit != all'
                , 'mit != rl'
                , 'mit != call4'
                , 'rl != call4'
                ]
        showImages(images, 3, titles=titles)
        # In[22]:
        #convertBinaryMatToOrcaReadable(call4 > 0, "data/chr%02d_chr%02d_call4.edges"%(chr1, chr1))
        #convertBinaryMatToOrcaReadable(mit > 0, "data/chr%02d_chr%02d_mit.edges"%(chr1, chr1))
        #convertBinaryMatToOrcaReadable(rl > 0, "data/chr%02d_chr%02d_rl.edges"%(chr1, chr1))
        #convertBinaryMatToOrcaReadable(all > 0, "data/chr%02d_chr%02d_all.edges"%(chr1, chr1))

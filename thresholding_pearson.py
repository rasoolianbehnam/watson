
# coding: utf-8

# Importing some stuff

# In[1]:


import numpy as np
import cv2
from utility import *
import matplotlib.pyplot as plt
from iced import normalization
from iced import filter
import os


# In[2]:


def log(image):
    out = image * 1.
    out[np.where(out <= 0)] = 1
    return np.log(out)

def pearsonII(image):
    out = pearson(image)
    out[np.where(out <= 0)] = 0
    return out

def rel_error(m1 ,m2):
    diff = np.abs(m1 - m2)
    return diff / (np.abs(np.minimum(m1+100, m2+100)))

def print_statistics(m1, text = "", print_results=True):
    max = np.max(m1)
    min = np.min(m1)
    mean = m1.mean()
    median = np.percentile(m1, 50)
    std = m1.std()
    if (print_results):
        print("#-----------------------------------------------------------------------#")
        print("Stats for %s: "% text)
        print("max: %f | min: %f | median: %f | mean: %f | std: %f"%(max, min, median, mean, std))
        print("#-----------------------------------------------------------------------#")
    return max, min, mean, std

def t_test(m1, m2):
    diff = m1 - m2
    n = diff.shape[0] * diff.shape[1]
    std = np.sqrt(np.sum(diff * diff) / (n - 1))
    t0 = diff.mean() * np.sqrt(n) / std
    return t0

def threshold_within_std(m1, k):
    min, max, mean, std = print_statistics(m1, print_results=False)
    return (m1 > mean - k * std) * (m1 < mean + k * std)


# Initiating matrices

# In[129]:

chr1 = 22
chr2 = chr1
print "chromosome 1: %s, chromosome 2: %s"%(chr1, chr2)
#print observed_mit_original.shape

mit = readMat("../data/extracted_all_contact_MIT/chr_%02d_%02d_500kb.mat"%              (chr1, chr2), "\t")
call4 = readMat("../data/extracted_all_contact_call4/chr_%02d_%02d_500kb.mat"%                (chr1, chr2), "\t")
rl = readMat("../data/extracted_all_contact_RL/chr_%02d_%02d_500kb.mat"%(chr1, chr2), "\t")


# In[130]:


all = readMat("../data/extracted_all_contact_ALL/chr_%02d_%02d_500kb.mat"%(chr1, chr2), "\t")


# In[131]:


n0, m0 = np.shape(mit)
n1, m1 = np.shape(call4)
n2, m2 = np.shape(rl)
n3, m3 = np.shape(all)

n = np.min([n0, m0, n1, m1, n2, m2, n3, m3])
mit = mit[:n, :n]
call4 = call4[:n, :n]
rl = rl[:n, :n]
all = all[:n, :n]


# In[132]:


mit = pearson(scn(normalize(mit)))
call4 = pearson(scn(normalize(call4)))
rl = pearson(scn(normalize(rl)))
print call4.shape


# In[133]:


all = pearson(scn(normalize(rl)))


# In[134]:


min, max, mean_mit, std_mit = print_statistics(mit, "mit")
min, max, mean_rl, std_rl = print_statistics(rl, "rl")
min, max, mean_call4, std_call4 = print_statistics(call4, "call4")
min, max, mean_all, std_all = print_statistics(all, "all")


# In[135]:


#images = [
#        (call4) > 0
#        , (mit) > 0
#        , (rl) > 0
#        , (all) > 0
#        ]
#showImages(images, 1)


# In[136]:


#images = [
#        np.log(call4+1)
#        , np.log(mit+1)
#        , np.log(rl+1)
#        , np.log(all+1)
#        ]
#showImages(images, 1)


# In[137]:


convertBinaryMatToOrcaReadable(call4 > 0, "data/chr%02d_chr%02d_call4.edges"%(chr1, chr1))
convertBinaryMatToOrcaReadable(mit > 0, "data/chr%02d_chr%02d_mit.edges"%(chr1, chr1))
convertBinaryMatToOrcaReadable(rl > 0, "data/chr%02d_chr%02d_rl.edges"%(chr1, chr1))
convertBinaryMatToOrcaReadable(all > 0, "data/chr%02d_chr%02d_all.edges"%(chr1, chr1))


# In[143]:


os.system('Rscript rscript.r %s'%chr1)


# In[144]:


graphlets_mit = readMat("data/chr%02d_chr%02d_mit.graphlets"%(chr1, chr1), delimiter=" ").astype('uint32')
graphlets_all = readMat("data/chr%02d_chr%02d_all.graphlets"%(chr1, chr1), delimiter=" ").astype('uint32')
graphlets_rl = readMat("data/chr%02d_chr%02d_rl.graphlets"%(chr1, chr1), delimiter=" ").astype('uint32')
graphlets_call4 = readMat("data/chr%02d_chr%02d_call4.graphlets"%(chr1, chr1), delimiter=" ").astype('uint32')
print(graphlets_mit.shape)
print(graphlets_all.shape)
print(graphlets_rl.shape)
print(graphlets_call4.shape)


# In[145]:


mit_all_corr =     graphlet_correlational_distance(graphlets_mit, graphlets_all)


# In[146]:


mit_mit_corr =     graphlet_correlational_distance(graphlets_mit, graphlets_mit)


# In[147]:


mit_call4_corr =     graphlet_correlational_distance(graphlets_mit, graphlets_call4)


# In[148]:


mit_rl_corr =     graphlet_correlational_distance(graphlets_mit, graphlets_rl)


# In[149]:


#images = [ mit_mit_corr
#          , mit_all_corr
#         , mit_rl_corr
#         , mit_call4_corr]
#showImages(images, 2)


# In[150]:


print_statistics(mit_mit_corr)
print_statistics(mit_all_corr)
print_statistics(mit_rl_corr)
print_statistics(mit_call4_corr)


# In[151]:


print_statistics(np.diagonal(mit_mit_corr))
print_statistics(np.diagonal(mit_all_corr))
print_statistics(np.diagonal(mit_rl_corr))
print_statistics(np.diagonal(mit_call4_corr))


# In[152]:


graphlets = {'mit' : graphlets_mit
            , 'all' : graphlets_all
            , 'rl' : graphlets_rl
            , 'call4' : graphlets_call4}
pair_orbits(graphlets, chr1)


# In[153]:


for orbit in range(73):
    file_name = "data/chr%02d_orbit%02d.paired.csv"%(chr1, orbit)
    command = "java -jar mine.jar %s -allPairs"%file_name
    os.system(command)


# In[154]:


pairwise_hic = {}
for key in graphlets:
    pairwise_hic = {}
    for key in graphlets:
        pairwise_hic[key] = {}
for orbit in range(73):
    var1, var2, mic = get_mic_from_file        ("data/chr%02d_orbit%02d.paired.csv,allpairs,cv=0.0,B=n^0.6,Results.csv"         %(chr1, orbit))
    n = len(var1)
    for i in range(n):
        pairwise_hic[var1[i]][var2[i]] = mic[i]
        pairwise_hic[var2[i]][var1[i]] = mic[i]


# In[155]:


pairwise_hic


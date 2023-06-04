#!/usr/bin/python3

#%matplotlib widget
import os
from typing import List, Tuple

import cv2
import imageio 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
#from flt_reader import read_flt_image
from tqdm import tqdm
import json

from numba import jit

def readfile(filename):

    try:

        filename = str(filename.encode("unicode_escape").decode('utf-8'))

        image = imageio.imread(filename)

    except:

        print("Datei " , filename , " kann nicht geoeffnet werden.")

        raise 

        sys.exit()

    return image


images = [0] * 9 # initialize array

exposure_times = [ 7, 98 , 197 , 499 , 998 , 1997 , 4999 , 9997 , 20001 , 40002 , 99999 , 149999 , 199998 , 299997 , 500002 , 899998 , 1.2e+06 , 1.8e+06 , 2.5e+06 , 3e+06 , 7e+06 , 1e+07]

#my load ldr images
def readldr(): 
    for i in range(len(images)): #j, i in enumerate([2,5,8]):
        readoutpath = f"ExpCurve_{i}.exr"
        pic = readfile(readoutpath)
        #import pdb;pdb.set_trace()
        pic[pic > 1] = 1
        #pic = (pic * (2**12 - 1)).astype(np.uint64)
        #print(f"pic min = {np.min(pic)} , pic max = {np.max(pic)}")
        images[i] = pic
        
# @jit(nogil=True)
# def fillocc():
#     size = 4095
#     bar = tqdm(range(size), desc="calculation")
    
#     for i in bar:
#         for pic in images:
#                 selected_pixels = pic == i 
#                 occurences[i] += np.sum(selected_pixels)
                
#                 selected_pixels[:] = False
                
#     print("0 has: " , occurences[0])    
#     print("2048 has: " , occurences[2047])
#     print("4095 has: " , occurences[4095])         
readldr()

#occurences = np.zeros(2**12, dtype=np.int64)
#fillocc()

fig, axs = plt.subplots(1, 1, figsize=[4, 3])

#axs.plot(occurences)

(n, bins, patches) = plt.hist(np.array(images).reshape([-1]), histtype='stepfilled', bins= 2**12)

for elem in n:
    print(elem)


axs.set_xlabel('Intensity Value')
axs.set_ylabel('Count')
fig.savefig("histogram.png")

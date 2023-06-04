#!/usr/bin/python3

#%matplotlib widget
import os
from typing import List, Tuple

import cv2
import imageio 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp
#from flt_reader import read_flt_image
from tqdm import tqdm
import json

from numba import jit





with open("theta_H.binary", "rb") as f:
    buffer = f.read()
    theta_H = np.frombuffer(buffer, np.int32)

with open("theta_diff.binary", "rb") as f:
    buffer = f.read()
    theta_diff = np.frombuffer(buffer, np.int32)

with open("phi_diff.binary", "rb") as f:
    buffer = f.read()
    phi_diff = np.frombuffer(buffer, np.int32)

with open("indices.binary", "rb") as f:
    buffer = f.read()
    indices = np.frombuffer(buffer, np.int32)
    
with open("indices_main_arc.binary", "rb") as f:
    buffer = f.read()
    indices_main_arc = np.frombuffer(buffer, np.int32)

with open("indices_hemi.binary", "rb") as f:
    buffer = f.read()
    indices_hemi = np.frombuffer(buffer, np.int32)      

with open("wi_angles.binary", "rb") as f:
    buffer = f.read()
    wi_angles = np.frombuffer(buffer, np.float32)#.reshape([-1, 2])    

with open("locations_.binary", "rb") as f:
    buffer = f.read()
    locs = np.frombuffer(buffer, np.float32).reshape([-1, 6])


#to90 = list(range(0, 30))
#to180 = list(range(0,120))
#toind = list(range(0,120* 30 * 30))

to90 = list(range(0, 90))
to180 = list(range(0,180))
toind = list(range(0,180* 90 * 90))

print("Phi-Diff = " , phi_diff)
print("Theta-Diff = " , theta_diff)
print("Theta-H = " , theta_H)


fig, ax = plt.subplots()
#plt.hist(phi_diff, density=True, bins=180)
ax.set(xlim=(0, len(to180)), ylim=(0, 600000))
ax.plot(to180, phi_diff)
fig.savefig('phi_diff.png')
plt.title("phi_diff")
fig2 , ax2 = plt.subplots()

#plt.hist(theta_diff, density=True, bins=90)
ax2.set(xlim=(0, len(to90)), ylim=(0, 6000000))
plt.xlabel('Bins')
plt.ylabel('Density')
ax2.plot(to90, theta_diff)
plt.title("theta diff")
fig2.savefig('theta_diff.png')


fig3 , ax3 = plt.subplots()
#plt.hist(theta_H, density=True, bins=90)
ax3.set(xlim=(0, len(to90)), ylim=(0, 600000))
ax3.plot(to90, theta_H)
fig3.savefig('theta_H.png')
plt.title("theta_H")

fig4 , ax4 = plt.subplots()
#plt.hist(theta_H, density=True, bins=90)
#ax4.bar(toind, indices, 0.1 , color='r',log=True)
ax4.set(xlim=(0, len(toind)), ylim=(0, 1500))
ax4.plot(toind, indices)
plt.xlabel('Bins')
plt.ylabel('Density')
plt.title("Indices")
fig4.savefig('indices.png')



#print("Max before rounding: " ,max(wi_angles))
#print(wi_angles[0], ", " , wi_angles[10000] , ", " , wi_angles[91050231])

#@jit(nopython=True)
def round2dcplc(dc, ls):
    func = lambda x: round(x,dc)
    return [list(map(func, i)) for i in ls]
    


#wi_rounded = round2dcplc(3,wi_angles)

#print("Max after rounding: " ,max(wi_angles))
#print(wi_angles[0], ", " , wi_angles[10000] , ", " , wi_angles[91050231])

#wi_idx = np.linspace(0, 3.14, num=314, retstep=true, step=0.01)
#wi = [0] * 314
#idx = 0

#for elem in wi_idx:
#    wi[idx] = wi_rounded.count(elem)
#    idx++


#wo , wi = zip(*wi_angles)
#plt.hist(theta_H, density=True, bins=90)
#ax4.bar(toind, indices, 0.1 , color='r',log=True)
#ax5.set(xlim=(0, 1.58), ylim=(0, 1)) #list(range(0,len(wi_angles))),
#plt.scatter(range(0.00, 1.58, 0.01), wi_angles)
#ax5.plot(wi_angles) range(0.00, 1.58, 0.01)

#plt.scatter(wo , wi , alpha=0.01)

#ax5.set(xlim=(0, 3.14), ylim=(0, 1))
#ax5.set(xlim=(0, 3.14))
#ax5.plot(wi_idx, wi)

#fig5.savefig('wi_angles.png')
#plt.title("wi_angles")

fig5 , ax5 = plt.subplots()

# Create a density plot
plt.hist(wi_angles, density=True, bins=900)

# Add labels and title
plt.xlabel('Angle')
plt.ylabel('Density')
plt.title('Density Plot of Angles')
fig5.savefig('wi_angles.png')

def getangle(vec1,vec2):
    dot_prod = np.dot(vec1, vec2)

    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    cos_angle = dot_prod / (magnitude1 * magnitude2)
    return np.arccos(cos_angle)

    


X1, Y1, Z1, U1, V1, W1= zip(*locs)

ax5.scatter(-0.0247951131, -0.0882998854, 0.957063437, marker='o', color="red")

center = -(np.array([0.0832537, 0.0557775, -0.994966]))

vecarr = []# * (len(X1) * (len(X1) - 1))
idx = 0

for elem in X1:
    temp = np.array([-U1[idx], -V1[idx], -W1[idx]])
    idy = 0
    for elem2 in X1:
        
        temp2 = np.array([-U1[idy], -V1[idy], -W1[idy]])
        
        if((temp[0] == temp2[0]) and (temp[1] == temp2[1]) and (temp[2] == temp2[2])):
            
            idy = idy + 1
            
            continue
        
        angle = getangle(temp,temp2)
        if (angle > 0.3):
            idy = idy + 1
            continue
        
        vecarr.append(angle)
        idy = idy + 1
    
    idx = idx + 1

print(len(vecarr))
#print(vecarr)   
print("Max-Angle= " , max(vecarr))
print("Min-Angle= " , min(vecarr))
print("Avg-Angle= " , (sum(vecarr) / len(vecarr)))
plt.show()

wh = np.count_nonzero(indices)
h = np.count_nonzero(indices_hemi)
ma =np.count_nonzero(indices_main_arc)

print(f'Number of indexes filled by whole hemi array: {wh}')
print(f'Number of indexes filled by hemi array: {h}')
print(f'Number of indexes filled by main arc arrays: {ma}')


filled_in_main_arc_only = np.count_nonzero(np.logical_and(indices_main_arc, np.logical_not(indices_hemi)))


filled_in_hemi_only = np.count_nonzero(np.logical_and(indices_hemi, np.logical_not(indices_main_arc)))


common_indexes = np.count_nonzero(np.logical_and(indices_hemi, indices_main_arc))
print(f'Number of common indexes filled in both arrays: {common_indexes}')

print(f'Number of indexes filled by hemi only: {filled_in_hemi_only}')

print(f'Number of indexes filled by main arc only: {filled_in_main_arc_only}')




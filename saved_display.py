# -*- coding: utf-8 -*-
"""
Older display script. Not used by trn.py.
"""

# import some librariesw
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import saved_sf2 as sf
import model as mm

#%% read multi-channel dataset

plt.clf()
plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, 1.0))
subplot_i=1
fig_i = 1

trnSmri, trnInv, trnAtb, trnSmap, encode, mb_slices = sf.getData(0.1)
tstInv = np.reshape(trnInv[0], (sf.shapez, sf.shapex, sf.shapey, sf.acceleration))
inpInv = np.abs(tstInv)#.clip(-32767, 32767)

image = np.zeros((sf.shapez * sf.acceleration, sf.shapex, sf.shapey))
assert mb_slices.shape == (sf.acceleration, sf.shapez)
for z in range(sf.shapez):
    for acc in range(sf.acceleration):
        image[mb_slices[acc, z]] = inpInv[z, ..., acc]

for z in range(4, 60, 4):
    plt.subplot(4, 4, subplot_i)
    subplot_i += 1
    plot(sf.normalize01(image[z]))
    plt.axis('off')
    plt.title(f"z={z}")

    #plt.savefig("shiftcombo" + str(fig_i) + ".png")
    #plt.clf()
plt.show()
plt.clf()

subplot_i = 1
for x in range(5, 85, 5):
    plt.subplot(4, 4, subplot_i)
    subplot_i += 1
    plot(sf.normalize01(image[:, x]))
    plt.axis('off')
    plt.title(f"x={x}")

    #plt.savefig("shiftcombo" + str(fig_i) + ".png")
    #plt.clf()
plt.show()

# -*- coding: utf-8 -*-

# import some librariesw
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import supportingFunctions as sf
import model as mm

#%% read multi-channel dataset

plt.clf()
plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, 1.0))
subplot_i=1
fig_i = 1

smriFilename = "/home/jhutter3/data/Exam10085/T1_MPRAGEDicom/T1_MPRAGE_Series0003_T1_MPRAGE_20220224133646_3%toepi.nii"
acqFilename = "/home/jhutter3/data/Exam10085/Series10/kSpace/kspace.h5"
print(acqFilename, smriFilename)

tstSmri, tstInv, tstB, tstSmap, encode, mb_slices = sf.getData(smriFilename, acqFilename, [0], True)
print(tstInv.shape)
image = np.abs(sf.r2c(tstInv))
print(image.shape)

for z in range(3, 60, 8):
    plt.subplot(4, 4, subplot_i)
    subplot_i += 1
    plot(sf.normalize01(image[z]))
    plt.axis('off')
    plt.title(f"fMRI z={z}")

    plt.subplot(4, 4, subplot_i)
    subplot_i += 1
    plot(sf.normalize01(tstSmri[0, z]))
    plt.axis('off')
    plt.title(f"sMRI z={z}")

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

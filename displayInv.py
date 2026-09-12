# -*- coding: utf-8 -*-

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

smriFilename = "/data/projects/jhutter/Exam9992/T1_MPRAGEDicom/T1_MPRAGE_Series0003_T1_MPRAGE_20220128094618_3%toepi.nii"
acqFilename = "/data/projects/jhutter/Exam9992/Series10/kSpace/kspace.h5"

print(acqFilename, smriFilename)

tstSmri, tstInv, tstB, tstSmap, encode, mb_slices = sf.getData(smriFilename, acqFilename, [0], True)
print(tstInv.shape)
image = np.abs(sf.r2c(tstInv))
print(image.shape)

print(image.max())

print("inv mean is", np.mean(image, where=image > 0.000001))
print("inv std is", np.std(image, where=image > 0.000001))
print("smri mean is", np.mean(tstSmri, where=tstSmri > 0.000001))
print("smri std is", np.std(tstSmri, where=tstSmri > 0.000001))

if False:
    plt.hist([np.abs(0.15 - np.abs(0.15 - image.flatten())), np.abs(0.15 - np.abs(0.15 - tstSmri.flatten()))], label=['inv', 'smri'], bins=20, range=(0., 0.4))
    plt.legend(loc='upper right')
    plt.title("Histogram of nonzero values")
    plt.show()
    plt.clf()

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

#   for x in range(90):
#        for y in range(90):
#            if image[z, x, y]> 0.001:
#                print((x, y, z), image[z, x, y], tstSmri[0, z, x, y])

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

# -*- coding: utf-8 -*-
"""
This is the training code to train the model as described in the following article:

MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

Paper dwonload  Link:     https://arxiv.org/abs/1712.02862

This code solves the following optimization problem:

    argmin_x ||Ax-b||_2^2 + ||x-Dw(x)||^2_2

 'A' can be any measurement operator. Here we consider parallel imaging problem in MRI where
 the A operator consists of FFT and coil sensitivity maps.

Dw(x): it represents the residual learning CNN.

Here is the description of the parameters that you can modify below.

epochs: how many times to pass through the entire dataset

nLayer: number of layers of the convolutional neural network.
        Each layer will have filters of size 3x3. There will be 64 such filters
        Except at the first and the last layer.

gradientMethod: AG, MG is gone. set MG for 'manual gradient' of conjuagate gradient (CG) block
                as discussed in section 3 of the above paper. Set it to AG if
                you want to rely on the tensorflow to calculate gradient of CG.

K: it represents the number of iterations of the alternating strategy as
    described in Eq. 10 in the paper.  Also please see Fig. 1 in the above paper.
    Higher value will require a lot of GPU memory. Set the maximum value to 20
    for a GPU with 16 GB memory. Higher the value more is the time required in training.

sigma: the standard deviation of Gaussian noise to be added in the k-space

batchSize: You can reduce the batch size to 1 if the model does not fit on GPU.

Output:

After running the code the output model will be saved in the subdirectory 'savedModels'.
You can give the name of the generated ouput directory in the tstDemo.py to
run the newly trained model on the test data.


@author: Hemant Kumar Aggarwal
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

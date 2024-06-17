# -*- coding: utf-8 -*-
"""
This is the training code to train the model as described in the following article:

MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

Paper dwonload  Link:     https://arxiv.org/abs/1712.02862

This code solves the following optimization problem:

    argmin_x ||Ax-b||_2^2 + ||x-Dw(x)||^2_2

 'A' can be any measurement operator. Here we consider parallel imaging problem in MRI where
 the A operator consists of undersampling mask, FFT, and coil sensitivity maps.

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
import supportingFunctions as sf
import model as mm

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

#--------------------------------------------------------------

tstOrg,tstAtb,tstCsm,tstMask=sf.getTestingData()
modelFile = "savedModels/25Apr_0819pm_3L_1K_2E_/model.keras"
#modelFile = "savedModels/25Apr_0918pm_3L_2K_2E_/model.keras"
modelFile = "savedModels/25Apr_0927pm_3L_1K_50E_/model.keras"
modelFile = "savedModels/25Apr_1032pm_5L_1K_50E_/model.keras"
modelFile = "savedModels/26Apr_1204pm_5L_1K_50E_/model.keras"
modelFile = "savedModels/30Apr_1232pm_10L_1K_50E_/model.keras"
modelFile = "savedModels/02May_1210pm_10L_1K_2E_/model.keras"
modelFile = "savedModels/02May_0351pm_5L_2K_5E_/model.keras"
modelFile = "savedModels/16May_0947pm_5L_1K_50E_/model.keras"
modelFile = "savedModels/17May_0814am_5L_1K_50E_/model.keras"
modelFile = "savedModels/17May_0914am_5L_1K_20E_/model.keras"
modelFile = "savedModels/17May_0922am_5L_1K_20E_/model.keras"
modelFile = "savedModels/17May_0929am_5L_1K_20E_/model.keras"
modelFile = "savedModels/17May_0952am_5L_1K_20E_/model.keras"
modelFile = "savedModels/17May_1007am_5L_1K_20E_/model.keras"
modelFile = "savedModels/17May_1030am_5L_1K_20E_/model.keras"

rec = []
with tf.keras.utils.custom_object_scope({'ConjugateGradientLayer': mm.ConjugateGradientLayer}, {'mi_customloss': mm.mi_customloss}):
    loadedModel = tf.keras.models.load_model(modelFile)
    rec = loadedModel.predict([tstCsm, tstMask, tstAtb])

normOrg = sf.normalize01( np.abs(tstOrg) )
normAtb = sf.normalize01( np.abs(sf.r2c(tstAtb))) 
normRec = sf.normalize01( np.abs(rec) )

psnrAtb = sf.myPSNR(normOrg,normAtb)
psnrRec = sf.myPSNR(normOrg,normRec)

print ('*****************')
print ('  ' + 'Noisy ' + 'Recon')
print ('  {0:.2f} {1:.2f}'.format(psnrAtb,psnrRec))
print ('*****************')

#%% Display the output images
plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, .8))
plt.clf()
plt.subplot(141)
plot(np.fft.fftshift(tstMask[0]))
plt.axis('off')
plt.title('Mask')
plt.subplot(142)
plot(normOrg)
plt.axis('off')
plt.title('Original')
plt.subplot(143)
plot(normAtb)
plt.title('Input, PSNR='+str(psnrAtb.round(2))+' dB' )
plt.axis('off')
plt.subplot(144)
plot(normRec)
plt.title('Output, PSNR='+ str(psnrRec.round(2)) +' dB')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
plt.show()

print ('*************************************************')

#%%


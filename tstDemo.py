# -*- coding: utf-8 -*-
"""
Leftover demo script. Training is trn.py; this file is not used there.
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

tstOrg,tstAtb,tstCsm=sf.getData()
if len(tstOrg) > 1:
    print("tst data can only handle 1 scan and 1 timepoint within that scan")
    assert False

modelFile = "savedModels/17May_1030am_5L_1K_20E_/model.keras"

rec = []
with tf.keras.utils.custom_object_scope({'ConjugateGradientLayer': mm.ConjugateGradientLayer}, {'mi_customloss': mm.mi_customloss}):
    loadedModel = tf.keras.models.load_model(modelFile)
    rec = loadedModel.predict([tstCsm, tstMask, tstAtb])

tstOrg = np.squeeze(tstOrg)
tstAtb = np.squeeze(tstAtb)
rec = np.squeeze(rec)

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


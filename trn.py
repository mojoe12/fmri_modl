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
#% SET THESE PARAMETERS CAREFULLY
nLayers=5
epochs=20
batchSize=30
K=1
sigma=0.01
restoreWeights=False
restoreFromModel='savedModels/02May_0345pm_5L_1K_5E_/model.keras'
#%% to train the model with higher K values  (K>1) such as K=5 or 10,
# it is better to initialize with a pre-trained model with K=1.
if K>1:
    restoreWeights=True

#if restoreWeights:

#--------------------------------------------------------------------------
#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='savedModels/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+ \
 str(nLayers)+'L_'+str(K)+'K_'+str(epochs)+'E_'

if not os.path.exists(directory):
    os.makedirs(directory)

csmT = tf.keras.Input(dtype=tf.complex64,shape=(12,256,232),name='csm')
maskT = tf.keras.Input(dtype=tf.complex64,shape=(256,232),name='mask')
atbT = tf.keras.Input(dtype=tf.float32,shape=(256,232,2),name='atb')

out = mm.makePhysicsAggarwalModel(atbT,csmT,maskT,nLayers,K)
model = tf.keras.Model(inputs=[csmT, maskT, atbT], outputs=out)
print(model.summary())

#%% read multi-channel dataset
trnOrg,trnAtb,trnCsm,trnMask = sf.getData('training')
print("org images data shape is ", trnOrg.shape)
print("csm data shape is ", trnCsm.shape)
print("max output real intensity is ", np.max(np.real(trnOrg)))

#%% creating the dataset

trnIn = tf.data.Dataset.from_tensor_slices((trnCsm, trnMask, trnAtb))
trnOut = tf.data.Dataset.from_tensor_slices(trnOrg)
trnData = tf.data.Dataset.zip((trnIn, trnOut)).cache().batch(1) # need to batch by 1 to get the first dimension

#%% make training model

#loss = tf.reduce_mean(tf.reduce_sum(tf.pow(predT-orgT, 2),axis=0)) # this is our loss function

#%% training code

print ('training started at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('parameters are: Epochs:',epochs,' BS:',batchSize,'nSamples:',trnOrg.shape[0])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.), loss=mm.mi_customloss)

if restoreWeights:
    model.load_weights(restoreFromModel, by_name=True)

csv_logger = tf.keras.callbacks.CSVLogger(directory + "/training.log")
history = model.fit(
    trnData,
    shuffle=True,
    batch_size=batchSize,
    epochs=epochs,
    verbose=2,
    callbacks=[csv_logger],
)

end_time = time.time()
print ('Training completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))

modelSaveFile = directory + "/model.keras"
model.save(modelSaveFile)
print ('model saved to', modelSaveFile)



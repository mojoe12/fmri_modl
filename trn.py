# -*- coding: utf-8 -*-
"""
Train an SMS-EPI SENSE unfold from GE k-space.

The forward model is coil sensitivity maps times the multiband / PE encoding
matrix. Training optionally cycles a residual CNN through that unfold and
uses mutual information against a T1 already in EPI space.

Edit smriFilenames and acqFilenames below. Knobs: epochs, nLayers, K,
nTimepoints, minibatchSize. Checkpoints go under savedModels/.
"""

# import some librariesw
import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import random
import saved_sf2 as sf
import model as mm
import h5py as h5

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

#--------------------------------------------------------------
#% SET THESE PARAMETERS CAREFULLY
nLayers=5
epochs=3
K=3
sigma=0.01 # this is irrelevant for fmri rn
displayTest=True
nTimepoints = 10
nExams = 1
minibatchSize = 3

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

csmShape = (sf.shapez,sf.shapex, sf.n_channels, sf.shapey * sf.acceleration, 2)
bShape = (sf.shapez, sf.shapex, sf.acq_shapey, sf.n_channels, 2)
smriShape = (sf.shapez,sf.shapex,sf.shapey,sf.acceleration)
csmT = tf.keras.Input(dtype=tf.float32, shape=csmShape, name='csm')
bT = tf.keras.Input(dtype=tf.float32, shape=bShape, name='b')

#%% read multi-channel dataset
#smriFilename = /data/projects/jhutter/testing/Exam9992/Series12/dicomImages/dicomImages_NOT_DIAGNOSTIC__CUFF2_20220128094618_1200f.nii").get_fdata()
#smriFilename = "/data/projects/jhutter/testing/Pt_Exam9992/T1_MPRAGE_Series0003/T1_MPRAGE_Series0003_T1_MPRAGE_20220128094618_3.nii"
#smriFilename = "/data/projects/jhutter/testing/skullstrip/s64004_T1_MPRAGE_20220128094618_3%toepi.nii"
#smriFilename = nib.load("/data/projects/jhutter/testing/withskull/T1_original%toepi.nii")
#acqFilename = "/home/qluo/software/orchestra-sdk-lastest/orchestra-sdk/test/kSpace/kspace.h5"
#acqFilename = "/data/projects/jhutter/testing/Exam9992/Series12/H5Out/hdf5.out"
#acqFilename = "/data/projects/jhutter/testing/Exam9992/Series12Full/kSpace/kspace.h5"

smriFilenames = ["/data/projects/jhutter/testing/skullstrip/s64004_T1_MPRAGE_20220128094618_3%toepi.nii"]
assert len(smriFilenames) == nExams
acqFilenames = ["/data/projects/jhutter/testing/Exam9992/Series12Full/kSpace/kspace.h5"]
assert len(acqFilenames) == nExams

tstTimepoints = [0, sf.n_volumes-1]
tstSmri, tstInv, tstB, tstCsm, trnEncode, trnMbSlices = sf.getData(smriFilenames[0], acqFilenames[0], tstTimepoints, True)
valid_slices = np.repeat(np.where(trnMbSlices < 60, 1., 0.).astype(np.float32).T, sf.shapey, axis=1)

print("smri images data shape is ", tstSmri.shape, "dtype is", tstSmri.dtype)
print("b images data shape is ", tstB.shape, "dtype is", tstB.dtype)
print("csm data shape is ", tstCsm.shape, "dtype is", tstCsm.dtype)
print("output magnitude intensity has max", np.max(tstSmri), "mean", np.mean(tstSmri))

assert tstB.shape[1:] == bShape
assert tstCsm.shape[1:] == csmShape
assert tstSmri.shape[1:] == smriShape
assert trnEncode.shape == (sf.acq_shapey, sf.acceleration * sf.shapey, 2)

#%% creating the dataset

trnSmri = np.zeros((nExams,) + smriShape)
trnCsm = np.zeros((nExams,) + csmShape)
for exam_index in range(nExams):
    outSmri, outInv, outB, outCsm, outEncode, outMbSlices = sf.getData(smriFilenames[exam_index], acqFilenames[exam_index], [], False)
    trnSmri[exam_index] = outSmri
    trnCsm[exam_index] = outCsm

#trnIn = tf.data.Dataset.from_tensor_slices((trnCsm, trnB)).batch(1)
#trnOut = tf.data.Dataset.from_tensor_slices(trnSmri).batch(1)
#trnData = tf.data.Dataset.zip((trnIn, trnOut)).cache().batch(1) # need to batch by 1 to get the first dimension

#n_samples = len(trnB)
#trnData = tf.data.Dataset.from_generator(
#    lambda: ((trnCsm, trnB), trnSmri),
#    output_types  = ((tf.complex64, tf.float32), tf.float32),
    #output_shapes = ((list(csmShape).insert(0, None), list(bShape).insert(0, None)), list(smriShape).insert(0, None))
#)

#%% make training model


print ('training started at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('parameters are: Epochs:',epochs,' MBS:',minibatchSize,'nSamples:',nTimepoints)

#%% training code

csv_logger = tf.keras.callbacks.CSVLogger(directory + "/training.log")
histories = []

modelSaveFile = directory + "/model.keras"

sample_indices_list = []
for exam_index in range(nExams):
    for timepoint_index in range(nTimepoints):
        sample_indices_list.append((exam_index, timepoint_index * (sf.n_volumes // nTimepoints)))

def fitModel(this_K, restoreWeights, encode):
    out = mm.makePhysicsAggarwalModel(bT,csmT,nLayers,this_K, tf.convert_to_tensor(encode), tf.convert_to_tensor(valid_slices))
    model = tf.keras.Model(inputs=[csmT, bT], outputs=out)
    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5, clipvalue=10000.), loss=mm.mi_customloss)

    if len(restoreWeights) > 0:
        restoreWeights_index = 0
        for layer in model.layers:
            if "conv2d" in layer.name or "activation" in layer.name:
                layer.set_weights(restoreWeights[restoreWeights_index])
                print("for layer", layer, "used index", restoreWeights_index, "of old weights")
                restoreWeights_index = (restoreWeights_index + 1) % len(restoreWeights)

    for epoch_index in range(epochs):
        random.shuffle(sample_indices_list)
        for minibatch_index in range(minibatchSize):
            exam_index, timepoint = sample_indices_list[minibatch_index]
            trnB = sf.getFmriDataOnly(h5.File(acqFilenames[exam_index]), [timepoint])
            with tf.device('/cpu:0'):
                xcsm = tf.convert_to_tensor(trnCsm[exam_index:exam_index+1])
                xb = tf.convert_to_tensor(trnB[:])
                y=tf.convert_to_tensor(trnSmri[exam_index:exam_index+1])
            history = model.fit(
                x=[xcsm, xb],
                y=y,
                #shuffle=True,
                batch_size=1,
                epochs=epoch_index+1,
                initial_epoch=epoch_index,
                verbose=2,
                callbacks=[csv_logger],
            )
            histories.append(history)
        print(f"Finished {epoch_index+1}/{epochs} epochs")

    #model.save(modelSaveFile) # come back to this one
    #print ('model saved to', modelSaveFile)
    return model

model = fitModel(1 if K > 0 else 0, [], trnEncode)

#%% to train the model with higher K values  (K>1) such as K=5 or 10,
# it is better to initialize with a pre-trained model with K=1.
if K>1:
    weights_by_layer = []
    for layer in model.layers:
        if "conv2d" in layer.name or "activation" in layer.name:
            weights_by_layer.append(layer.get_weights())
    del model
    model = fitModel(K, weights_by_layer, trnEncode)

end_time = time.time()
print ('Training completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))

if displayTest:

    rec_start_time = time.time()
    rec = model.predict([tstCsm, tstB[-1:]])
    print('Model prediction on one volume took', time.time() - rec_start_time)

    print ('Dumping reconstructed image from file', acqFilenames[0], "timepoint", sf.n_volumes-1)

    h5out = h5py.File('Reconstruction_output_exam_0.h5', 'w')
    h5out.create_dataset('timepoint_' + str(sf.n_volumes-1), data=rec)
    h5out.close()

    tstSmri = tstSmri[-1]
    tstInv = np.reshape(sf.r2c(tstInv), tstSmri.shape)
    tstRec = np.reshape(sf.r2c(rec[0]), tstSmri.shape)

    normSmri = sf.normalize01( np.abs(tstSmri) )
    normInv = sf.normalize01( np.abs(tstInv)) 
    normRec = sf.normalize01( np.abs(tstRec) )

    psnrInv = sf.myPSNR(normSmri,normInv)
    psnrRec = sf.myPSNR(normSmri,normRec)


    #%% Display the output images
    plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, .8))
    pictured_z, pictured_accel = 0, 4
    plt.clf()
    plt.subplot(131)
    plot(normSmri[pictured_z, :, :, pictured_accel])
    plt.axis('off')
    plt.title('Original')
    plt.subplot(132)
    plot(normInv[pictured_z, :, :, pictured_accel])
    plt.title('Input, PSNR='+str(psnrInv.round(2))+' dB' )
    plt.axis('off')
    plt.subplot(133)
    plot(normRec[pictured_z, :, :, pictured_accel])
    plt.title('Output, PSNR='+ str(psnrRec.round(2)) +' dB')
    plt.axis('off')
    plt.show()

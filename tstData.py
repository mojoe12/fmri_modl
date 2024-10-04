import os,time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import random
import supportingFunctions as sf
import model as mm
import h5py as h5
import glob

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        tf.config.experimental.set_memory_growth(gpu, False)
  except RuntimeError as e:
    print(e)

#--------------------------------------------------------------
#% SET THESE PARAMETERS CAREFULLY
nLayers=5
epochs=3
K=3
sigma=0.01 # this is irrelevant for fmri rn
nTimepoints = 1
csmShape = (sf.shapez,sf.shapex, sf.n_channels, sf.shapey * sf.acceleration, 2)
bShape = (sf.shapez, sf.shapex, sf.acq_shapey, sf.n_channels, 2)
smriShape = (sf.shapez,sf.shapex,sf.shapey,sf.acceleration)
csmT = tf.keras.Input(dtype=tf.float32, shape=csmShape, name='csm')
bT = tf.keras.Input(dtype=tf.float32, shape=bShape, name='b')


smriFilenames = glob.glob("/home/jhutter3/data/*/Series3/*T1_MPRAGE*%toepi.nii")
acqFilenames = glob.glob("/home/jhutter3/data/*/*/kSpace/kspace.h5")
assert len(smriFilenames) == len(acqFilenames)
print(acqFilenames)

tstTimepoint = 0
tstSmri, tstInv, tstB, tstCsm, trnEncode, trnMbSlices = sf.getData(smriFilenames[0], acqFilenames[0], [tstTimepoint], True)
valid_slices = np.repeat(np.where(trnMbSlices < 60, 1., 0.).astype(np.float32).T, sf.shapey, axis=1)

print("smri images data shape is ", tstSmri.shape, "dtype is", tstSmri.dtype)
print("b images data shape is ", tstB.shape, "dtype is", tstB.dtype)
print("csm data shape is ", tstCsm.shape, "dtype is", tstCsm.dtype)
print("output magnitude intensity has max", np.max(tstSmri), "mean", np.mean(tstSmri))

assert tstB.shape[1:] == bShape
assert tstCsm.shape[1:] == csmShape
assert tstSmri.shape[1:] == smriShape
assert trnEncode.shape == (sf.acq_shapey, sf.acceleration * sf.shapey, 2)

def fitModel(this_K, restoreWeights, encode):
    out = mm.makePhysicsAggarwalModel(bT,csmT,nLayers,this_K, tf.convert_to_tensor(encode), tf.convert_to_tensor(valid_slices))
    model = tf.keras.Model(inputs=[csmT, bT], outputs=out)
    print(model.summary())
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5., clipvalue=10000.), loss=mm.mi_customloss)

    if len(restoreWeights) > 0:
        restoreWeights_index = 0
        num_trainable_layers = 0
        for layer in model.layers[1:]:
            if len(layer.trainable_weights) > 0:
                if "p_inv_layer" in layer.name:
                    assert num_trainable_layers == 0 # assert that this one went first!
                layer.set_weights(restoreWeights[restoreWeights_index])
                print("for layer", layer.name, "used index", restoreWeights_index, "of old weights")
                restoreWeights_index = 1 + num_trainable_layers % (len(restoreWeights) - 1)
                num_trainable_layers += 1
        if len(restoreWeights) > 1:
            assert (num_trainable_layers - 1) % (len(restoreWeights) - 1) == 0
    return model

modelSavedFile = "/home/jhutter3/fmri_modl/savedModels/24Sep_0232pm_5L_3K_3E_/results.h5"
f = h5.File(modelSavedFile)
print(f.keys())
weights_by_layer = []
for layer_index in range(len(f['layer_names'])):
    layer_name = f['layer_names'][layer_index].decode('ascii')
    assert layer_index > 0 or "p_inv_layer" in layer_name
    weight_list = []
    f_layer = f["weights_" + layer_name]
    num_keys = len(f_layer.keys())
    for sublayer_index in range(len(f_layer.keys())):
        weight_list.append(f_layer[str(sublayer_index)])
    weights_by_layer.append(weight_list)

model = fitModel(K, weights_by_layer, trnEncode)

rec_start_time = time.time()
rec = model.predict([tstCsm, tstB[-1:]])
print('Model prediction on one volume took', time.time() - rec_start_time, 'seconds')

print ('Dumping reconstructed image from file', acqFilenames[0], "timepoint", tstTimepoint)

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


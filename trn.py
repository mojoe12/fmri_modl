"""
This is the training code to train the model as described in the following article:

Author: Joseph Hutter

Inherited from MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

This code solves the following optimization problem:

    argmin_x ||Ax-b||_2^2 + ||x-Dw(x)||^2_2

'A' can be any measurement operator. Here we consider parallel imaging problem in MRI where
the A operator consists of FFT and coil sensitivity maps.
Dw(x): it represents the residual learning CNN.

Here is the description of the parameters that you can modify below.

nLayers: number of layers of the convolutional neural network.
         Each layer will have filters of size 3x3. There will be 64 such filters
         Except at the first and the last layer.

K: it represents the number of iterations of the alternating strategy as
   described in Eq. 10 in the paper.  Also please see Fig. 1 in the above paper.
   Higher value will require a lot of GPU memory. Set the maximum value to 20
   for a GPU with 16 GB memory. Higher the value more is the time required in training.

Output:
After running the code the output model will be saved in the subdirectory 'savedModels'.
You can give the name of the generated ouput directory in the tstDemo.py to
run the newly trained model on the test data.


@author: Joseph Hutter
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
import supportingFunctions as sf
import model as mm
import h5py as h5
import glob
import argparse

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
parser = argparse.ArgumentParser(prog="trn", description="Trains algorithm")
parser.add_argument('--smriFilespec', default='/home/jhutter3/data/*/T1_MPRAGEDicom/*T1_MPRAGE*%toepi.nii')
parser.add_argument('--acqFilespec', default='/home/jhutter3/data/*/*/kSpace/kspace.h5')
parser.add_argument('--nExams', type=int, default=300)
parser.add_argument('--nLayers', type=int, default=4)
parser.add_argument('--nBlocks', type=int, default=1)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--nTimepoints', type=int, default=1)
parser.add_argument('--displayTest', action='store_true')
parser.add_argument('--testExam', default="Exam9992Test", required=True)
parser.add_argument('--testTimepoint', type=int, default=0)
parser.add_argument('--minibatchSize', type=int, default=8)
parser.add_argument('--useAggarwal', type=bool, default=True)
parser.add_argument('--numHBuckets', type=int, default=30)
parser.add_argument('--betaMult', type=float, default=50)
args = parser.parse_args()

nExams = min(args.nExams, args.epochs * args.minibatchSize)

#--------------------------------------------------------------------------
#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='savedModels/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+ \
 str(args.nLayers)+'L_'+str(args.nBlocks)+'K_'+str(args.epochs)+'E_'+ \
 str(nExams)+'N_'+str(args.minibatchSize)+'M_'+str(args.betaMult)+'B_'+ \
 ("Aggarwal" if args.useAggarwal else "UNet")

if not os.path.exists(directory):
    os.makedirs(directory)

csmShape = (sf.shapez,sf.shapex, sf.n_channels, sf.shapey * sf.acceleration, 2)
bShape = (sf.shapez, sf.shapex, sf.acq_shapey, sf.n_channels, 2)
smriShape = ((sf.shapez - 1) * sf.acceleration,sf.shapex,sf.shapey)
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

smriAllFilenames = glob.glob(args.smriFilespec)
acqAllFilenames = glob.glob(args.acqFilespec)
assert len(acqAllFilenames) >= nExams
acqFilenames, smriFilenames = [], []
for acqF in acqAllFilenames:
    assert "Exam" in acqF
    acqF_exam_start = acqF.find("Exam")
    acqF_exam_end = acqF.find("/", acqF_exam_start)
    exam_string = acqF[acqF_exam_start:acqF_exam_end]
    smriFound = False
    for smriF in smriAllFilenames:
        if exam_string + '/' in smriF:
            assert not smriFound
            if exam_string == args.testExam:
                acqTestFilename = acqF
                smriTestFilename = smriF
            elif len(acqFilenames) < nExams:
                acqFilenames.append(acqF)
                smriFilenames.append(smriF)
            smriFound = True
    if not smriFound:
        print(acqF)
    assert smriFound
print(acqFilenames)
print(smriFilenames)
assert len(acqFilenames) == nExams
assert len(smriFilenames) == nExams

tstSmri, tstInv, tstB, tstCsm, trnEncode, trnMbSlices = sf.getData(smriTestFilename, acqTestFilename, [args.testTimepoint], True)

print("smri images data shape is ", tstSmri.shape, "dtype is", tstSmri.dtype)
print("b images data shape is ", tstB.shape, "dtype is", tstB.dtype)
print("csm data shape is ", tstCsm.shape, "dtype is", tstCsm.dtype)
print("output magnitude intensity has max", np.max(tstSmri), "mean", np.mean(tstSmri))

assert tstB.shape[1:] == bShape
assert tstCsm.shape[1:] == csmShape
assert tstSmri.shape[1:] == smriShape
assert trnEncode.shape == (sf.acq_shapey, sf.acceleration * sf.shapey, 2)
assert trnMbSlices.shape[1:] == ((sf.shapez - 1) * sf.acceleration, sf.shapez, 1, sf.shapey * sf.acceleration, sf.shapey, 1)

#%% creating the dataset

trnSmri = np.zeros((nExams,) + smriShape, dtype=np.float32)
trnCsm = np.zeros((nExams,) + csmShape, dtype=np.float32)

for exam_index in range(nExams):
    outSmri, outInv, outB, outCsm, outEncode, outMbSlices = sf.getData(smriFilenames[exam_index], acqFilenames[exam_index], [], False)
    trnSmri[exam_index] = outSmri
    trnCsm[exam_index] = outCsm

print("smri magnitudes have min", trnSmri.min(), "mean", trnSmri.mean(), "max", trnSmri.max())

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
print ('parameters are: Epochs:',args.epochs,' MBS:',args.minibatchSize,'nSamples:',nExams)

#%% training code

modelSaveFile = directory + "/model.keras"

sample_indices_list = []
for exam_index in range(nExams):
    for timepoint_index in range(args.nTimepoints):
        sample_indices_list.append((exam_index, timepoint_index * (sf.n_volumes // args.nTimepoints)))

with tf.device('/cpu:0'):
    xcsmTest = tf.convert_to_tensor(tstCsm[-1:0])
    xbTest = tf.convert_to_tensor(tstB[-1:0])
    yTest = tf.convert_to_tensor(tstSmri[-1:0])

def fitModel(this_nBlocks, restoreWeights, encode, sliceMatrix):
    out = mm.makePhysicsModel(bT,csmT,args.nLayers,this_nBlocks, args.useAggarwal, tf.convert_to_tensor(encode), tf.convert_to_tensor(sliceMatrix))
    model = tf.keras.Model(inputs=[csmT, bT], outputs=out)
    print(model.summary())
    my_loss = mm.LossCalculator(args.numHBuckets, trnSmri.max(), args.betaMult)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10., clipvalue=500.), loss=my_loss.mi_customloss)
    histories = []
    test_losses = []
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

    for epoch_index in range(args.epochs):
        #csv_logger = tf.keras.callbacks.CSVLogger(directory + "/training.log", append=epoch_index>0)
        random.shuffle(sample_indices_list)
        for minibatch_index in range(min(args.minibatchSize, len(sample_indices_list))):
            exam_index, timepoint = sample_indices_list[minibatch_index]
            trnB = sf.getFmriDataOnly(h5.File(acqFilenames[exam_index]), [timepoint])
            with tf.device('/cpu:0'):
                xcsm = tf.convert_to_tensor(trnCsm[exam_index:exam_index+1])
                xb = tf.convert_to_tensor(trnB[:])
                y=tf.convert_to_tensor(trnSmri[exam_index:exam_index+1])
            #tf.profiler.experimental.start('logdir')
            trn_loss = model.train_on_batch(
                x=[xcsm, xb],
                y=y)
                #shuffle=True,
                #batch_size=1,
                #epochs=epoch+1,
                #initial_epoch=epoch,
                #verbose=1,
                #callbacks=[csv_logger],
            #)
            #epoch += 1
            #tf.profiler.experimental.stop()
            histories.append((acqFilenames[exam_index], timepoint, trn_loss))
        test_loss = model.test_on_batch(x=[tstCsm, tstB[-1:]], y=tstSmri)
        test_losses.append(test_loss)
        print(f"Finished {epoch_index+1}/{args.epochs} epochs. Test loss is {test_loss:.2e}")

    return model, histories, test_losses

model, histories, test_losses = fitModel(1 if args.nBlocks > 0 else 0, [], trnEncode, trnMbSlices)

#%% to train the model with higher K values  (K>1) such as K=5 or 10,
# it is better to initialize with a pre-trained model with K=1.
if args.nBlocks>1:
    weights_by_layer = []
    for layer in model.layers:
        if len(layer.trainable_weights) > 0:
            weights_by_layer.append(layer.get_weights())
    del model
    model, histories, test_losses = fitModel(K, weights_by_layer, trnEncode, trnMbSlices)

end_time = time.time()
print ('Training completed in', ((end_time - start_time) / 60), 'minutes')
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))

h5out = h5.File(directory + "/results.h5", 'w')
trainable_names = []
for layer in model.layers:
    if len(layer.trainable_weights) > 0:
        trainable_names.append(layer.name)
        layer_weights = layer.get_weights()
        subgroup = h5out.create_group("weights_" + layer.name)
        for layer_weight_index in range(len(layer_weights)):
            # technically nowaedays oyu can set track_order but i didnt want to assume people have that here
            subgroup.create_dataset(str(layer_weight_index), data=layer_weights[layer_weight_index])
h5_layernames = h5out.create_dataset("layer_names", (len(trainable_names)), dtype=h5.string_dtype())
for layer_index in range(len(trainable_names)):
    h5_layernames[layer_index] = trainable_names[layer_index]

h5_filenames = h5out.create_dataset("training_files", (len(histories)), dtype=h5.string_dtype())
h5_filenames.attrs['smriFilespec'] = args.smriFilespec
h5_timepoints = h5out.create_dataset("training_timepoints", (len(histories)), dtype=int)
h5_timepoints.attrs['nTimepoints'] = args.nTimepoints
h5_trnLosses = h5out.create_dataset("training_losses", (len(histories)), dtype=float)
h5_trnLosses.attrs['betaMult'] = args.betaMult
h5_trnLosses.attrs['numHBuckets'] = args.numHBuckets
for history_index in range(len(histories)):
    file, tp, trn_loss = histories[history_index]
    h5_filenames[history_index] = file
    h5_timepoints[history_index] = tp
    h5_trnLosses[history_index] = trn_loss
h5_testLosses = h5out.create_dataset("test_losses", data=test_losses)
h5_testLosses.attrs['acqFilename'] = acqTestFilename
h5_testLosses.attrs['smriFilename'] = smriTestFilename
h5_testLosses.attrs['testTimepoint'] = args.testTimepoint

rec_start_time = time.time()
rec = model.predict([tstCsm, tstB[-1:]])

print(np.max(np.abs(sf.r2c(rec))))
print('Model prediction on one volume took', time.time() - rec_start_time, 'seconds')

print ('Dumping reconstructed image from file', acqTestFilename, "timepoint", args.testTimepoint)

h5out.create_dataset('reconstruction_exam_0_timepoint_' + str(args.testTimepoint), data=rec)

tstSmri = tstSmri[-1]
tstInv = np.reshape(sf.r2c(tstInv), tstSmri.shape)
tstRec = np.reshape(sf.r2c(rec[0]), tstSmri.shape)

normSmri = sf.normalize01( np.abs(tstSmri) )
normInv = sf.normalize01( np.abs(tstInv))
normRec = sf.normalize01( np.abs(tstRec) )

psnrInv = sf.myPSNR(normSmri,normInv)
psnrRec = sf.myPSNR(normSmri,normRec)

for pictured_z in range(3, 60, 8):
    #%% Display the output images
    plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, .8))
    plt.clf()
    plt.subplot(131)
    plot(normSmri[pictured_z, :, :])
    plt.axis('off')
    plt.title('Original')
    plt.subplot(132)
    plot(normInv[pictured_z, :, :])
    plt.title('Input, PSNR='+str(psnrInv.round(2))+' dB' )
    plt.axis('off')
    plt.subplot(133)
    plot(normRec[pictured_z, :, :])
    plt.title('Output, PSNR='+ str(psnrRec.round(2)) +' dB')
    plt.axis('off')
    plt.savefig(directory + "/fig_" + args.testExam + "_z" + str(pictured_z) + ".png")
    if args.displayTest:
        plt.show()
    plt.clf()

for pictured_x in range(5, 85, 10):
    #%% Display the output images
    plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, .8))
    plt.clf()
    plt.subplot(131)
    plot(normSmri[:, pictured_x, :])
    plt.axis('off')
    plt.title('Original')
    plt.subplot(132)
    plot(normInv[:, pictured_x, :])
    plt.title('Input, PSNR='+str(psnrInv.round(2))+' dB' )
    plt.axis('off')
    plt.subplot(133)
    plot(normRec[:, pictured_x, :])
    plt.title('Output, PSNR='+ str(psnrRec.round(2)) +' dB')
    plt.axis('off')
    plt.savefig(directory + "/fig_" + args.testExam + "_x" + str(pictured_x) + ".png")
    if args.displayTest:
        plt.show()
    plt.clf()


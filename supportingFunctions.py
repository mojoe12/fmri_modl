"""
Created on Aug 6th, 2018

This file contains some supporting functions used during training and testing.

@author:Hemant
"""
import time
import numpy as np
import h5py as h5

#%%
def div0( a, b ):
    """ This function handles division by zero """
    c=np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return c

def normalize01(img):
    """
    Normalize the image between o and 1
    """
    if len(img.shape)==3:
        nimg=len(img)
    else:
        nimg=1
        r,c=img.shape
        img=np.reshape(img,(nimg,r,c))
    img2=np.empty(img.shape,dtype=img.dtype)
    for i in range(nimg):
        img2[i]=div0(img[i]-img[i].min(),img[i].ptp())
        #img2[i]=(img[i]-img[i].min())/(img[i].max()-img[i].min())
    return np.squeeze(img2).astype(img.dtype)

#%%
def np_crop(data, shape=(320,320)):

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]

#%%

def myPSNR(org,recon):
    """ This function calculates PSNR between the original and
    the reconstructed     images"""
    mse=np.sum(np.square( np.abs(org-recon)))/org.size
    psnr=20*np.log10(org.max()/(np.sqrt(mse)+1e-10 ))
    return psnr


#%% Here I am reading the dataset for training and testing from dataset.hdf5 file

def getData(trnTst='testing',num=100,sigma=.01):
    #num: set this value between 0 to 163. There are total testing 164 slices in testing data
    print('Reading the data. Please wait...')
    filename='dataset.hdf5' #set the correct path here
    #filename='/Users/haggarwal/datasets/piData/dataset.hdf5'

    with h5.File(filename) as f:
        if trnTst=='training':
            org,csm,mask=f['trnOrg'][:],f['trnCsm'][:],f['trnMask'][:]
        else:
            org,csm,mask=f['tstOrg'][num],f['tstCsm'][num],f['tstMask'][num]
            na=np.newaxis
            org,csm,mask=org[na],csm[na],mask[na]
    print('Successfully read the data from file!')
    print('Now doing undersampling....')
    atb=generateUndersampled(org,csm,mask,sigma)
    print('Successfully undersampled data!')
    atb=c2r(atb)
    print("atb shape now is ", atb.shape)
    return org,atb,csm,mask

#Here I am reading one single image from  demoImage.hdf5 for testing demo code

def getTestingData():
    print('Reading the data. Please wait...')
    filename='demoImage.hdf5' #set the correct path here
    with h5.File(filename,'r') as f:
        org,csm,mask=f['tstOrg'][:],f['tstCsm'][:],f['tstMask'][:]

    print('Successfully read the data from file!')
    print('Now doing undersampling....')
    atb=generateUndersampled(org,csm,mask,sigma=.01)
    print("atb shape now is ", atb.shape)
    atb=c2r(atb)
    print("atb shape now is ", atb.shape)
    print('Successfully undersampled data!')
    return org,atb,csm,mask

def getFMRIData():
    import nibabel as nib
    import ismrmrd
    f = ismrmrd.Dataset('/data/projects/jhutter/ismrmrd/ge_to_ismrmrd/sampleData/test70.h5', 'dataset', False)
    acqi = f.read_acquisition(0)
    #fMRI = 



#%%
def piA(x,csm,mask,nrow,ncol,ncoil):
    """ This is a the A operator as defined in the paper"""
    ccImg=np.reshape(x,(nrow,ncol) )
    coilImages=np.tile(ccImg,[ncoil,1,1])*csm;
    kspace=np.fft.fft2(coilImages)/np.sqrt(nrow * ncol)
    if len(mask.shape)==2:
        mask=np.tile(mask,(ncoil,1,1))
    res=kspace[mask!=0]
    return res

def piAt(kspaceUnder,csm,mask,nrow,ncol,ncoil):
    """ This is a the A^T operator as defined in the paper"""
    temp=np.zeros((ncoil,nrow,ncol),dtype=np.complex64)
    if len(mask.shape)==2:
        mask=np.tile(mask,(ncoil,1,1))

    temp[mask!=0]=kspaceUnder
    img=np.fft.ifft2(temp)*np.sqrt(nrow*ncol)
    coilComb=np.sum(img*np.conj(csm),axis=0).astype(np.complex64)
    #coilComb=coilComb.ravel();
    return coilComb

def generateUndersampled(org,csm,mask,sigma=0.):
    nSlice,ncoil,nrow,ncol=csm.shape
    atb=np.empty(org.shape,dtype=np.complex64)
    for i in range(nSlice):
        A  = lambda z: piA(z,csm[i],mask[i],nrow,ncol,ncoil)
        At = lambda z: piAt(z,csm[i],mask[i],nrow,ncol,ncoil)

        sidx=np.where(mask[i].ravel()!=0)[0]
        nSIDX=len(sidx)
        noise=np.random.randn(nSIDX*ncoil,)+1j*np.random.randn(nSIDX*ncoil,)
        noise=noise*(sigma/np.sqrt(2.))
        y=A(org[i]) + noise
        atb[i]=At(y)
    return atb


#%%
def r2c(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    if inp.dtype=='float32':
        dtype=np.complex64
    else:
        dtype=np.complex128
    out=np.zeros( inp.shape[0:2],dtype=dtype)
    out=inp[...,0]+1j*inp[...,1]
    return out

def c2r(inp):
    """  input img: row x col in complex64
    output image: row  x col x2 in float32
    """
    if inp.dtype=='complex64':
        dtype=np.float32
    else:
        dtype=np.float64
    out=np.zeros( inp.shape+(2,),dtype=dtype)
    out[...,0]=inp.real
    out[...,1]=inp.imag
    return out


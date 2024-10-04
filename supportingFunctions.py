"""
Created on Aug 6th, 2018

This file contains some supporting functions used during training and testing.

@author:Hemant
"""
import time
import numpy as np
import h5py as h5
import skimage
import tensorflow as tf
import nibabel as nib

n_channels = 32
shapex = 90
shapey = 90
acq_shapey = 66
shapez = 11
acceleration = 6
fov_shift = 3
n_volumes = 450

#%%
def div0( a, b ):
    """ This function handles division by zero """
    c=np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return c

def normalize01(img):
    """
    Normalize the image between 0 and 1
    """
    if len(img.shape)>=3:
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

def myPSNR(smri,recon):
    """ This function calculates PSNR between the original and
    the reconstructed     images"""
    mse=np.sum(np.square( np.abs(smri-recon)))/smri.size
    psnr=20*np.log10(smri.max()/(np.sqrt(mse)+1e-10 ))
    return psnr

def genSenseMap(image_coil, crop_smap):
    #print("image coil has shape", image_coil.shape)
    assert image_coil.shape == (shapex, shapey, shapez * acceleration, n_channels)
    im = np.sqrt(np.sum(image_coil * np.conj(image_coil), axis=3))
    smap = image_coil / np.expand_dims(im, axis=3)
    se = skimage.morphology.disk(radius=2)
    mask = np.zeros_like(im, dtype=np.uint8)
    for z in range(shapez * acceleration):
        im_temp = im[:, :, z]
        im_temp /= np.max(np.abs(im_temp))
        labels = im_temp > crop_smap
        remove_isolated = skimage.morphology.remove_small_objects(labels, 200) # changed
        mask[:, :, z] = skimage.morphology.binary_dilation(remove_isolated, se)
    return np.squeeze(smap * np.expand_dims(mask, 3))

def calcFftZ():
    ftz = np.zeros((acq_shapey, acceleration), dtype=np.complex64)
    assert fov_shift > 1
    #cap_blip_start_cal = fov_shift // 2
    #cal_blips = np.remainder(range(cap_blip_start_cal, cap_blip_start_cal + fov_shift), fov_shift)
    #shift_fractions = np.copysign((cal_blips - (fov_shift - 1) / 2) / fov_shift, fov_shift)
    assert fov_shift == 2 or fov_shift == 3
    if fov_shift == 2:
        shift_fractions = np.copysign(1, fov_shift) * np.array([-1/2, 0])
    elif fov_shift == 3:
        shift_fractions = np.copysign(1, fov_shift) * np.array([-1/2, -1/6, 1/6])
    assert shapey % 2 == 0
    ky = np.arange(-shapey // 2, -shapey // 2 + acq_shapey)
    idx = 0
    assert acceleration % fov_shift == 0
    z_indices = np.tile(np.arange(fov_shift), acceleration // fov_shift).astype(np.uint8)
    #print(shift_fractions)
    #print(z_indices)
    for acc in range(acceleration):
        z = z_indices[acc]
        ftz[:, acc] = np.exp(-2j * np.pi * ky * shift_fractions[z])
    return ftz

def getMtxYz(order):
    if order == 'C':
        encode_mtx_z = np.repeat(calcFftZ(), shapey, axis=1) #second axis is arrays of shapey
    else:
        encode_mtx_z = np.tile(calcFftZ(), (1, shapey)) #arrays of acceleration
    assert shapey % 2 == 0
    ky = np.arange(-shapey // 2, -shapey // 2 + acq_shapey) / shapey # shape acq_shapey
    y = np.fft.fftshift(np.arange(shapey)).astype(np.complex64) # shape shapey
    dfty_mtx = np.exp(-2j * np.pi * np.outer(ky, y)).astype(np.complex64) / np.sqrt(shapey) # shape acq_shapey x shapey
    if order == 'C':
        encode_mtx_y = np.tile(dfty_mtx, (1, acceleration)) # arrays of shapey
    else:
        encode_mtx_y = np.repeat(dfty_mtx, acceleration, axis=1) #arrays of acceleration
    return encode_mtx_y * encode_mtx_z

def matrix_trans(mtx):
    n_dims = len(mtx.shape)
    new_shape = [*range(n_dims - 2)] + [n_dims - 1, n_dims - 2]
    return np.transpose(mtx, new_shape)

def pinvLambda(mtx, sense_lam):
    ata = np.conj(matrix_trans(mtx)) @ mtx
    eye_shape = [1] * (len(mtx.shape) - 2) + [ata.shape[-2], ata.shape[-1]]
    ata += np.eye(ata.shape[-2], ata.shape[-1]) * sense_lam
    return np.linalg.inv(ata) @ np.conj(matrix_trans(mtx))

def getFmriDataOnly(acq_f, requested_timepoints):
    fmri_kspace = np.zeros((len(requested_timepoints), shapez, shapex, acq_shapey, n_channels), dtype=np.complex64) # we ifft this in a few lines
    for timepoint_index in range(len(requested_timepoints)):
        timepoint = requested_timepoints[timepoint_index]
        assert timepoint < n_volumes # either timepoint is too high or the above math is wrong!
        acq_tp = acq_f['kspace_data']['volume_' + str(timepoint+1)]
        acq_tp_complex = acq_tp[:]['real'] + 1j * acq_tp[:]['imag']
        #print ("input acq tp shape", acq_tp_complex.shape)
        fmri_kspace[timepoint_index] = np.transpose(acq_tp_complex, (0, 3, 2, 1))
    fmri_ifft = np.sqrt(acq_shapey) * np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(fmri_kspace, axes=2), axis=2), axes=2).astype(np.complex64) # ifft in x dim
    assert not np.any(np.isnan(fmri_ifft))
    return c2r(fmri_ifft)

def getData(smri_filename, acq_filename, requested_timepoints, do_inv, prev_mb_slice_matrix=None, sense_lam=0.1):
    print("Loading", smri_filename, acq_filename)
    smri_f = nib.load(smri_filename).get_fdata()
    acq_f = h5.File(acq_filename)
    mb_slices = acq_f['multiband_info']['multiband_slices'][:]
    mb_slice_matrix = np.zeros((1, (shapez - 1) * acceleration, shapez, 1, shapey * acceleration, shapey, 1), dtype=np.float32)
    smri = np.zeros((1, (shapez - 1) * acceleration, shapex, shapey), dtype=np.float32)
    for z in range(shapez):
        for acc in range(acceleration):
            slice_map = mb_slices[acc, z]
            if slice_map < 60:
                smri[0, slice_map, :, :] = smri_f[:, :, 59 - slice_map]
                for y in range(shapey):
                    mb_slice_matrix[0, slice_map, z, 0, y * acceleration + acc, y, 0] = 1.
    if prev_mb_slice_matrix is not None:
        assert mb_slice_mattrix == prev_mb_slice_matrix
    smri = np.rot90(smri, 2, axes=(2, 3)) # x,y dimensions
    assert not np.any(np.isnan(smri))
    # each brain volume is its own data point, but there are n_volumes * n_scans volumes
    craw_data = acq_f['raw_calibration_data']['image_space'][:]
    #craw_data = acq_f['raw_calibration']['surface_images'][:]
    #print("craw data raw shape", craw_data.shape)
    craw_complex = np.transpose(craw_data[:]['real'] + 1j * craw_data[:]['imag'], (3, 2, 0, 1))
    smap = genSenseMap(craw_complex[:], 0.05)
    smaps = np.zeros((1, shapez, shapex, n_channels, shapey * acceleration), dtype=np.complex64)
    for z in range(shapez): # for each of the 11 slices
        sms_slices = mb_slices[:, z]
        for x in range(shapex):
            smap_mtx = smap[x, :, sms_slices, :] # [shapey, acceleration, n_channels]
            smap_mtx = np.reshape(smap_mtx, (shapey * acceleration, n_channels), 'F').T # smap_mtx is now arrays of shapey
            smaps[0, z, x] = smap_mtx
    out_fmri = None
    out_inv = None
    out_encode = None
    if len(requested_timepoints) > 0:
        out_fmri = getFmriDataOnly(acq_f, requested_timepoints)
    if do_inv:
        assert out_fmri is not None
        #mtx_yz   is a (shapey * acceleration, acq_shapey) matrix
        #smap_mtx is a (shapey * acceleration, n_channels) matrix
        mtx_yz = getMtxYz('F')
        out_encode = c2r(mtx_yz)
        mtx_yz_pinv = np.reshape(np.linalg.pinv(mtx_yz), (1, 1, shapey * acceleration, 1, acq_shapey))
        smap_pinv = np.expand_dims(pinvLambda(smaps, sense_lam), axis=-1)
        fmri_b = np.expand_dims(np.expand_dims(r2c(out_fmri[-1]), 0), -3)
        inv = c2r(mtx_yz_pinv @ fmri_b @ smap_pinv)
        prod = np.tensordot(inv, mb_slice_matrix, axes=([1, 3], [2, 4]))
        out_inv = np.transpose(np.squeeze(prod), (2, 0, 3, 1))
    return smri,out_inv,out_fmri,c2r(smaps),out_encode,mb_slice_matrix

#%%
def r2c(inp):
    """  input img: row x col x 2 in float32
    output image: row  x col in complex64
    """
    assert inp.shape[-1] == 2
    if inp.dtype=='float32':
        dtype=np.complex64
    else:
        dtype=np.complex64
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

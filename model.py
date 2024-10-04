"""
This code will create the model described in our following paper
MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

Paper dwonload  Link:     https://arxiv.org/abs/1712.02862

@author: haggarwal
"""
import tensorflow as tf
import numpy as np
from os.path import expanduser
home = expanduser("~")
epsilon=1e-5
TFeps=tf.constant(1e-5,dtype=tf.float32)

# function c2r contatenate complex input as new axis two two real inputs
c2r=lambda x:tf.stack([tf.math.real(x),tf.math.imag(x)],axis=-1)
#r2c takes the last dimension of real input and converts to complex
r2c=lambda x:tf.complex(x[...,0],x[...,1])

def AggarwalLayer(x, lastLayer):
    """
    This function create a layer of CNN consisting of convolution, batch-norm,
    and ReLU. Last layer does not have ReLU to avoid truncating the negative
    part of the learned noise and alias patterns.
    """
    window_size = (3, 3, 3)
    n_channels = 2 if lastLayer else 64
    x = tf.keras.layers.Conv3D(filters=n_channels, kernel_size=window_size, padding='same')(x)
    x = tf.keras.layers.LayerNormalization()(x)

    if lastLayer:
        return x
    else:
        return tf.keras.layers.Activation(activation='relu')(x)

def makeDoubleConvBlock(x, n_filters):
    x = tf.keras.layers.Conv3D(filters=n_filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv3D(filters=n_filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x#tf.keras.layers.BatchNormalization()(x)

def makeDownsampleBlock(x, n_filters):
    f = makeDoubleConvBlock(x, n_filters)
    p = tf.keras.layers.MaxPool3D(2, padding='same')(f)
    p = tf.keras.layers.Dropout(0.3)(p)
    return f, p

def makeUpsampleBlock(x, conv_features, n_filters):
    x = tf.keras.layers.Conv3DTranspose(n_filters, 3, 2, padding='same')(x)
    x = tf.keras.layers.concatenate([x, conv_features])
    x = tf.keras.layers.Dropout(0.3)(x)
    x = makeDoubleConvBlock(x, n_filters)
    return x

def myPinvMatrix(mtx, input_lam):
    ata = tf.math.conj(tf.linalg.matrix_transpose(mtx)) @ mtx
    eye_shape = [1] * (len(ata.shape) - 2) + [ata.shape[-1], ata.shape[-1]]
    eye = tf.eye(ata.shape[-1], dtype=np.complex64)
    input_lam_complex = tf.complex(input_lam, tf.constant(0., dtype=np.float32))
    to_inv = ata + tf.broadcast_to(tf.reshape(eye, eye_shape), tf.shape(ata)) * input_lam_complex
    return tf.linalg.inv(to_inv)

def myPinv(mtx, input_lam):
    return myPinvMatrix(mtx, input_lam) @ tf.math.conj(tf.linalg.matrix_transpose(mtx))

class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self, csm, lam):
        with tf.name_scope('Ainit'):
            #self.csm = csm #tf.linalg.matrix_transpose(tf.math.conj(csm)) @ csm
            self.csm = tf.expand_dims(csm, -3)
            #self.csmPinv = tf.expand_dims(myPinv(csm, sense_lam), -1)
            self.lam = lam
    def myAtA(self, img):
        with tf.name_scope('AtA'):
            coilImage = tf.squeeze(tf.expand_dims(img, -2) @ tf.linalg.matrix_transpose(self.csm) @ tf.math.conj(self.csm))
            coilComb = coilImage + self.lam * img
        return coilComb

class ConjugateGradientLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.05, stddev=0.)
        self.lam = self.add_weight(shape=(), initializer=initializer, dtype=np.float32)
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    def call(self, b, z, csm):
        A = Aclass(csm, tf.complex(self.lam, tf.constant(0., dtype=np.float32)))
        rhs = r2c(b + self.lam * z)
        x = tf.zeros_like(rhs)
        r, p = rhs, rhs
        rTr = tf.reduce_sum(tf.math.conj(r) * r)
        for i in range(10): #acceleration is 6
            Ap = A.myAtA(p)
            alpha = rTr / tf.reduce_sum(tf.math.conj(p) * Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.reduce_sum(tf.math.conj(r) * r)
            beta = rTrNew / rTr
            p = r + beta * p
            rTr = rTrNew

        out = x #tf.squeeze(tf.squeeze(encodePinv @ tf.expand_dims(tf.linalg.matrix_transpose(x), -1), -1), -1)
        return out
        #return c2r(x)
        #Aop = tf.linalg.LinearOperatorIdentity(rhs.shape[2])# rhs.shape[3], is_selfadjoint=False, is_square=False)
        #return tf.linalg.experimental.conjugate_gradient(Aop, rhs, tol=1e-10, max_iter=10)

class PInvLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.)
        self.lam = self.add_weight(shape=(), initializer=initializer, dtype=np.float32)
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    def call(self, b, z, csmPinvLam, encodePinv, csm, slice_matrix):
        #print(b.shape, r2c(z).shape, csm.shape)
        rhs = r2c(b) + tf.linalg.matrix_transpose(tf.math.conj(csm @ r2c(self.lam * z)))
        inv = c2r(tf.squeeze(tf.squeeze(encodePinv @ tf.expand_dims(rhs, -3) @ csmPinvLam, -1), -1))
        #print(slice_matrix.shape, inv.shape)
        out = tf.squeeze(tf.squeeze(tf.squeeze(tf.tensordot(inv, slice_matrix, axes=([1, 3], [2, 4])), 3), 4), -1)
        #print("rhs shape is", rhs.shape, "out shape is", out.shape)
        return tf.transpose(out, (0, 3, 1, 4, 2))

def makePhysicsModel(b, csm, nLayers, K, useAggarwal, encode, sliceMatrix):
    with tf.name_scope('myModel'):
        encode_exp = tf.expand_dims(tf.expand_dims(r2c(encode), 0), 0)
        encodePinv = tf.expand_dims(myPinv(encode_exp, 0.001), -2)
        cg = PInvLayer()
        csmPinvLam = tf.expand_dims(myPinv(r2c(csm), cg.lam), -1)
        z = tf.zeros_like(c2r(tf.squeeze(tf.expand_dims(r2c(b), -3) @ csmPinvLam, -1)))
        x = cg(b, z, csmPinvLam, encodePinv, r2c(csm), sliceMatrix)
        for i in range(1,K+1):
            """
            This micro loop is the Dw block as defined in the Fig. 1 of the MoDL paper
            It creates an n-layer (nLay) residual learning CNN.
            Convolution filters are of size 3x3 and 64 such filters are there.
            nw: It is the learned noise
            dw: it is the output of residual learning after adding the input back.
            """
            z_img = x
            if useAggarwal:
                for j in np.arange(1,nLayers+1):
                    z_img = AggarwalLayer(z_img, j==nLayers)
            else:
                n_filters_initial = pow(2, nLayers + 1) # +1 because of real vs complex
                f1, z_img = makeDownsampleBlock(z_img, n_filters_initial)
                z_img = makeDoubleConvBlock(z_img, 2 * n_filters_initial)
                z_img = makeUpsampleBlock(z_img, f1, n_filters_initial)
            #mb_slice_matrix = np.zeros((1, (shapez - 1) * acceleration, shapez, 1, shapey * acceleration, shapey, 1), dtype=np.float32)
            #print(z_img.shape, sliceMatrix.shape)
            z_sliced = tf.squeeze(tf.squeeze(tf.squeeze(tf.tensordot(z_img + x, sliceMatrix, ([1, 2], [1, 5])), 3), 4), 5)
            z_trans = tf.transpose(z_sliced, (0, 3, 1, 4, 2))
            #print(z_trans.shape)
            encoded_z = encode_exp * tf.expand_dims(r2c(z_trans), -2) # (66 x 540) * (1 x 540)
            encoded_z_H = c2r(tf.linalg.matrix_transpose(tf.math.conj(encoded_z)))
            x = cg(b, encoded_z_H, csmPinvLam, encodePinv, r2c(csm), sliceMatrix)
    return x

def makePureAggarwalModel(b,csm,nLayers):
    x = b
    with tf.name_scope('myModel'):
        for j in np.arange(nLayers):
            x = AggarwalLayer(x, j+1 == nLayers)
    return r2c(x)

def makePureUNetModel(b,csm):
    x = b
    with tf.name_scope('myModel'):
        f1, x = makeDownsampleBlock(x, 64)
        #f2, x = makeDownsampleBlock(x, 128)
        #f3, x = makeDownsampleBlock(x, 256)
        x = makeDoubleConvBlock(x, 128)
        #f4, x = makeDownsampleBlock(x, 512)
        #x = makeDoubleConvBlock(x, 1024)
        #x = makeUpsampleBlock(x, f4, 512)
        #x = makeUpsampleBlock(x, f3, 256)
        #x = makeUpsampleBlock(x, f2, 128)
        x = makeUpsampleBlock(x, f1, 64)
        #x = tf.keras.layers.Conv2D(2, 1, padding='same', activation='softmax')(x)
    return r2c(x)

class LossCalculator:
    def __init__(self, num_hbuckets, max_intensity, beta_mult):
    #num_hbuckets = 30
    #max_intensity = 3000 # circumstantial
    # beta_mult = 50
        self.hbuckets = tf.convert_to_tensor(np.linspace(0.5, num_hbuckets - 0.5, num=num_hbuckets, dtype=np.float32) * max_intensity / num_hbuckets)
        self.beta = beta_mult / (max_intensity * max_intensity * 90 * 90 * 60) # didnt use real voxel counts because this is number is still tbd
        self.sigmoid_slope = num_hbuckets / max_intensity # we want to map (-3,3) of sigmoid input to one bucket

    def mse_customloss(self, y_true, y_pred):
        return self.beta * tf.reduce_mean(tf.square(tf.math.abs(r2c(y_pred)) - y_true))

    def mi_customloss(self, y_true, y_pred):
        f_sigmoids = tf.math.sigmoid(self.sigmoid_slope * (tf.expand_dims(tf.math.abs(r2c(y_pred)), -1) - self.hbuckets))
        s_sigmoids = tf.math.sigmoid(self.sigmoid_slope * (tf.expand_dims(y_true, -1) - self.hbuckets))
        sigmoid_products = tf.expand_dims(f_sigmoids * (1 - f_sigmoids), -1) * tf.expand_dims(s_sigmoids * (1 - s_sigmoids), -2)
        pfs = tf.reduce_sum(tf.reduce_sum(sigmoid_products, axis=1), axis=1)
        pf = tf.reduce_sum(pfs, axis=1)
        ps = tf.reduce_sum(pfs, axis=0)
        pf_ps = pf[:, None] * ps[None, :]
        return -tf.reduce_mean(pfs * tf.math.log(pfs / pf_ps)) + self.beta * tf.reduce_mean(tf.square(y_true - tf.math.abs(r2c(y_pred))))


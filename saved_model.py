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
    window_size = 3
    n_channels = 2 if lastLayer else 64
    x = tf.keras.layers.Conv2D(filters=n_channels, kernel_size=window_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if lastLayer:
        return x
    else:
        return tf.keras.layers.Activation(activation='relu')(x)

def makeDoubleConvBlock(x, n_filters):
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x#tf.keras.layers.BatchNormalization()(x)

def makeDownsampleBlock(x, n_filters):
    f = makeDoubleConvBlock(x, n_filters)
    p = tf.keras.layers.MaxPool2D(2, padding='same')(f)
    p = tf.keras.layers.Dropout(0.3)(p)
    return f, p

def makeUpsampleBlock(x, conv_features, n_filters):
    x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding='same')(x)
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
    def call(self, atb, z, csm):
        A = Aclass(csm, tf.complex(self.lam, tf.constant(0., dtype=np.float32)))
        rhs = r2c(atb + self.lam * z)
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
    def call(self, atb, z, csmPinv, encodePinv):
        rhs = r2c(atb + self.lam * z)
        out = tf.squeeze(tf.squeeze(r2c(encodePinv) @ tf.expand_dims(rhs, -3) @ r2c(csmPinv), -1), -1)
        print("rhs shape is", rhs.shape, "out shape is", out.shape)
        return c2r(out)

class LstSqLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.05, stddev=0.)
        self.lam = self.add_weight(shape=(), initializer=initializer, dtype=np.float32)
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    def call(self, atb, z, csm):
        rhs = r2c(atb + self.lam * z)
        #print(tf.linalg.matrix_transpose(rhs).dtype)
        #csmPinv = tf.expand_dims(myPinv(csm, 0.1), -1)
        #print("comparison, pinv middle shape is", (tf.expand_dims(rhs, -3) @ csmPinv).shape)
        ata = tf.math.conj(tf.linalg.matrix_transpose(csm)) @ csm
        eye_shape = [1] * (len(ata.shape) - 2) + [ata.shape[-1], ata.shape[-1]]
        eye = tf.eye(ata.shape[-1], dtype=np.complex64)
        input_lam_complex = tf.complex(self.lam, tf.constant(0., dtype=np.float32))
        to_inv = ata + tf.broadcast_to(tf.reshape(eye, eye_shape), tf.shape(ata)) * input_lam_complex
        print("to inv shape is", to_inv.shape, csm.shape, rhs.shape)
        atb_true = tf.expand_dims(rhs, -3) @ tf.expand_dims(tf.math.conj(tf.linalg.matrix_transpose(csm)), axis=-1)
        atb_reshaped = tf.squeeze(atb_true, -1)
        out = tf.linalg.lstsq(to_inv, atb_reshaped, fast=True)
        print("rhs shape is", rhs.shape, "out shape is", out.shape)
        return out

class LstSqLayer2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.05, stddev=0.)
        self.lam = self.add_weight(shape=(), initializer=initializer, dtype=np.float32)
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    def call(self, atb, z, csm):
        rhs = r2c(atb + self.lam * z)
        #print(tf.linalg.matrix_transpose(rhs).dtype)
        #csmPinv = tf.expand_dims(myPinv(csm, 0.1), -1)
        #print("comparison, pinv middle shape is", (tf.expand_dims(rhs, -3) @ csmPinv).shape)
        print(csm.shape, rhs.shape)
        s, u, v = tf.linalg.svd(csm)
        d = tf.cast(s, np.complex64) 
        input_lam_complex = tf.complex(0.1, tf.constant(0., dtype=np.float32))
        inp = v @ tf.linalg.diag(d ** 2) + v * input_lam_complex
        inp_2 = tf.linalg.inv(inp @ tf.math.conj(tf.linalg.matrix_transpose(v)))
        #print(inp.shape, inp_2.shape, v.shape, tf.linalg.diag(d).shape, tf.linalg.matrix_transpose(u).shape, tf.linalg.matrix_transpose(rhs).shape)
        out = inp_2 @ v @ tf.linalg.diag(d) @ tf.math.conj(tf.linalg.matrix_transpose(u)) @ tf.linalg.matrix_transpose(rhs)
        #print("rhs shape is", rhs.shape, "out shape is", out.shape)
        return out

def makePhysicsAggarwalModel(atb, csm, nLayers, K, encode, valid_slices):
    #cg = ConjugateGradientLayer()
    with tf.name_scope('myModel'):
        encodePinv = c2r(tf.expand_dims(tf.expand_dims(tf.expand_dims(myPinv(r2c(encode), 0.001), 0), 0), -2))
        cg = PInvLayer()
        csmPinv = c2r(tf.expand_dims(myPinv(r2c(csm), cg.lam), -1))
        z = tf.zeros_like(atb)
        x = cg(atb, z, csmPinv, encodePinv)
        for i in range(1,K+1):
            """
            This micro loop is the Dw block as defined in the Fig. 1 of the MoDL paper
            It creates an n-layer (nLay) residual learning CNN.
            Convolution filters are of size 3x3 and 64 such filters are there.
            nw: It is the learned noise
            dw: it is the output of residual learning after adding the input back.
            """
            for j in np.arange(1,nLayers+1):
                x = AggarwalLayer(x, j==nLayers)
            z = c2r(tf.linalg.matrix_transpose(r2c(csm) @ r2c(x))) # ((32 x 540) @ (540, 66)).T
            x = cg(atb, z, csmPinv, encodePinv)
    return c2r(r2c(x) * tf.expand_dims(valid_slices, 1))

def makePureAggarwalModel(atb,csm,nLayers):
    x = atb
    with tf.name_scope('myModel'):
        for j in np.arange(nLayers):
            x = AggarwalLayer(x, j+1 == nLayers)
    return r2c(x)

def makePureUNetModel(atb,csm):
    x = atb
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

def makePhysicsUNetModel(atb,csm,K):
    cg = ConjugateGradientLayer()
    x = atb
    for i in range(K):
        x = c2r(makePureUNetModel(x,csm))
        x = c2r(cg(atb, x, csm))
    return r2c(x)

def mse_customloss(y_true, y_pred):
    y_pred_reshape = tf.reshape(tf.math.abs(r2c(y_pred)), tf.shape(y_true))
    y_pred_norm = y_pred_reshape# / tf.math.reduce_max(y_pred_reshape)
    return tf.reduce_mean(tf.square(y_true - y_pred_norm))

num_hbuckets = 30
max_intensity = 0.1 # circumstantial
hbuckets = tf.convert_to_tensor(np.linspace(0.5, num_hbuckets - 0.5, num=num_hbuckets, dtype=np.float32) * max_intensity / num_hbuckets)
beta = 0.000005 #in order to equal the weight, you want 0.0000005
sigmoid_slope = 10 # 50 causes nan, 25 is circumstance

def mi_customloss(y_true, y_pred):
    y_pred_reshape = tf.reshape(tf.math.abs(r2c(y_pred)), tf.shape(y_true))
    f_sigmoids = tf.math.sigmoid(sigmoid_slope * (tf.expand_dims(tf.math.abs(y_pred_reshape), -1) - hbuckets))
    s_sigmoids = tf.math.sigmoid(sigmoid_slope * (tf.expand_dims(y_true, -1) - hbuckets))
    sigmoid_products = tf.expand_dims(f_sigmoids * (1 - f_sigmoids), -1) * tf.expand_dims(s_sigmoids * (1 - s_sigmoids), -2)
    pfs = tf.reduce_sum(tf.reduce_sum(sigmoid_products, axis=1), axis=1)
    pf = tf.reduce_sum(pfs, axis=1)
    ps = tf.reduce_sum(pfs, axis=0)
    pf_ps = pf[:, None] * ps[None, :]
    return -tf.reduce_mean(pfs * tf.math.log(pfs / pf_ps)) + beta * tf.reduce_mean(tf.square(y_true - tf.math.abs(y_pred_reshape)))


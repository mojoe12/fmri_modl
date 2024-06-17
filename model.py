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
    print("p shape after maxpool2d for", n_filters, "is", p.shape)
    p = tf.keras.layers.Dropout(0.3)(p)
    return f, p

def makeUpsampleBlock(x, conv_features, n_filters):
    x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding='same')(x)
    print("x shape after conv2dt for", n_filters, "is", x.shape)
    x = tf.keras.layers.concatenate([x, conv_features])
    x = tf.keras.layers.Dropout(0.3)(x)
    x = makeDoubleConvBlock(x, n_filters)
    return x

class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self, csm,mask,lam):
        with tf.name_scope('Ainit'):
            s = tf.shape(mask)
            self.nrow,self.ncol = s[0],s[1]
            self.pixels = self.nrow*self.ncol
            self.mask = mask
            self.csm = csm
            self.SF = tf.complex(tf.sqrt(tf.cast(self.pixels, tf.float32) ),0.)
            self.lam = lam
            #self.cgIter=cgIter
            #self.tol=tol
    def myAtA(self,img):
        with tf.name_scope('AtA'):
            coilImages = self.csm*img
            kspace = tf.signal.fft2d(coilImages)/self.SF
            temp = kspace*self.mask
            coilImgs = tf.signal.ifft2d(temp)*self.SF
            coilComb = tf.reduce_sum(coilImgs * tf.math.conj(self.csm), axis=1)
            coilComb = coilComb+self.lam*img
            #formula = ifft(fft(csm * img) / scal * mask) * scal * conj(csm) # as a correction to img
            #formula = csm * img * conj(csm)
        return coilComb

class ConjugateGradientLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        initializer = tf.keras.initializers.RandomNormal(mean=0.05, stddev=0.)
        self.lam = self.add_weight(shape=(), initializer=initializer)
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    def call(self, atb, rhs, csm, mask):
        A = Aclass(csm, mask, tf.complex(self.lam, 0.))
        rhs = r2c(atb + self.lam * rhs)
        x = tf.zeros_like(rhs)
        r, p = rhs, rhs
        rTr = tf.reduce_sum(tf.math.conj(r) * r)
        for i in range(0, 10):
            Ap = A.myAtA(p)
            alpha = rTr / tf.reduce_sum(tf.math.conj(p) * Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.reduce_sum(tf.math.conj(r) * r)
            beta = rTrNew / rTr
            p = r + beta * p
            rTr = rTrNew
        return x
        #return c2r(x)
        #Aop = tf.linalg.LinearOperatorIdentity(rhs.shape[2])# rhs.shape[3], is_selfadjoint=False, is_square=False)
        #return tf.linalg.experimental.conjugate_gradient(Aop, rhs, tol=1e-10, max_iter=10)

def makePhysicsModel(atb,csm,mask):
    cg = ConjugateGradientLayer()
    return cg(atb, atb, csm, mask)

def makePhysicsAggarwalModel(atb,csm,mask,nLayers,K):
    cg = ConjugateGradientLayer()
    x = atb
    with tf.name_scope('myModel'):
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
            x = c2r(cg(atb, x, csm, mask))
    return r2c(x)

def makePureAggarwalModel(atb,csm,mask,nLayers):
    x = atb
    with tf.name_scope('myModel'):
        for j in np.arange(nLayers):
            x = AggarwalLayer(x, j+1 == nLayers)
    return r2c(x)

def makePureUNetModel(atb,csm,mask):
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

def makePhysicsUNetModel(atb,csm,mask,K):
    cg = ConjugateGradientLayer()
    x = atb
    for i in range(K):
        x = c2r(makePureUNetModel(x,csm,mask))
        x = c2r(cg(atb, x, csm, mask))
    return r2c(x)

def mse_customloss(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.math.real(y_true) - tf.math.real(y_pred))+tf.square(tf.math.imag(y_true) - tf.math.imag(y_pred)))

num_hbuckets = 30
max_intensity = 1.3 # circumstantial
hbuckets = tf.convert_to_tensor(np.linspace(0.5, num_hbuckets - 0.5, num=num_hbuckets, dtype=np.float32) * max_intensity / num_hbuckets)
beta = 100000 #circumstantial
sigmoid_slope = 25 # 50 causes nan, 25 is circumstance

def mi_customloss(y_true, y_pred):
    f_sigmoids = tf.math.sigmoid(sigmoid_slope * (tf.expand_dims(tf.math.real(y_pred), -1) - hbuckets))
    s_sigmoids = tf.math.sigmoid(sigmoid_slope * (tf.expand_dims(tf.math.real(y_true), -1) - hbuckets))
    sigmoid_products = tf.expand_dims(f_sigmoids * (1 - f_sigmoids), -1) * tf.expand_dims(s_sigmoids * (1 - s_sigmoids), -2)
    pfs = tf.reduce_sum(tf.reduce_sum(sigmoid_products, axis=1), axis=1)
    pf = tf.reduce_sum(pfs, axis=1)
    ps = tf.reduce_sum(pfs, axis=0)
    pf_ps = pf[:, None] * ps[None, :]
    return -tf.reduce_mean(pfs * tf.math.log(pfs / pf_ps)) + beta * tf.reduce_mean(tf.square(tf.math.real(y_true) - tf.math.real(y_pred))+tf.square(tf.math.imag(y_true) - tf.math.imag(y_pred)))


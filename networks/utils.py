import numpy as np
import tensorflow as tf
import pickle

def vec2mat(vec,size):
    """
    Convert vector to matrix
    vec : (..., size[0]*size[1])
    mat : (..., size[0], size[1])
    """
    shape = tf.shape(vec)
    newShape = tf.concat([shape[:-1], size], axis=0)
    return tf.reshape(vec, newShape)

def mat2vec(vec):
    """
    Convert(flatten) matrix to vector
    vec : (..., size0*size1)
    mat : (..., size0, size1)
    """
    shape = tf.shape(mat)
    newShape = tf.concat([shape[:-2], shape[-1]*shape[-2]], axis=0)
    return tf.reshape(mat, newShape)

def lnorm(x,mu,Sig):
    """
    Compute logN(x; mu,Sig)
    x, mu, Sig : (..., d)
    """
    r = (x-mu)**2/Sig # (..., d)
    ldet_Sig = tf.reduce_sum(tf.log(Sig), axis=-1,keepdims=True) # (..., 1)
    return tf.reduce_sum(-0.5*r - 0.5*tf.log(2*np.pi), axis=-1, keepdims=True)  - 0.5*ldet_Sig  # (..., 1) 
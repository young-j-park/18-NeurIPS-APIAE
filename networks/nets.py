import numpy as np
import tensorflow as tf
import pickle
from .utils import *
     
class DynNet(object):
    """ Neural-net for the dynamics model: zdot=dz/dt=f(z) """
    def __init__(self, intermediate_units, n_z, n_u):
        self.intermediate_units = intermediate_units
        self.Layers = []
        
        self.n_z = n_z
        self.n_u = n_u

        # Construct the neural network
        for i, unit in enumerate(self.intermediate_units[:-2]):
            self.Layers.append(
                tf.layers.Dense(units=unit, activation=tf.nn.relu, name='DynLayer' + str(i)))  # fully-connected layer
        
        self.Layers.append(tf.layers.Dense(units=self.intermediate_units[-2], activation=tf.nn.softmax))  # fully-connected layer
        self.Layers.append(tf.layers.Dense(units=self.intermediate_units[-1], use_bias=False, name='DynLayerLast',                        
     kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=0.1)))
        
        # Below is for the later use of this network
        self.z_in = tf.placeholder(tf.float32, (None, n_z))
        A_out, b_out, sigma_out = self.compute_Ab_sigma(self.z_in)
        self.sigma_out = sigma_out
        self.zdot_out = tf.squeeze(A_out@tf.expand_dims(self.z_in,axis=-1) + b_out,axis=-1)
        
        # Below is for the initialization
        self.A_init = tf.placeholder(tf.float32, (n_z, n_z))
        self.b_init = tf.placeholder(tf.float32, (n_z, 1))
        self.sigma_init = tf.placeholder(tf.float32, (n_z, n_u))
        
        self.loss_A = tf.reduce_mean(tf.reduce_sum((A_out - tf.expand_dims(self.A_init,axis=0)) ** 2, axis=-1))
        self.loss_b = tf.reduce_mean(tf.reduce_sum((b_out - tf.expand_dims(self.b_init,axis=0)) ** 2, axis=-1))
        self.loss_sigma = tf.reduce_mean(tf.reduce_sum((sigma_out - tf.expand_dims(self.sigma_init,axis=0)) ** 2, axis=-1))
        self.loss = self.loss_A + self.loss_b + self.loss_sigma
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def compute_Ab_sigma(self, z):
        # Input :  z=(...,n_z)
        # Output: zdot=(...,n_z), sigma=(...,n_z,n_u)
        
        temp = z # z_2 dot = neural_net(z)
        for layer in self.Layers:
            temp = layer(temp)
        
        A, b, sigma = tf.split(temp, [self.n_z*self.n_z, self.n_z, self.n_z*self.n_u], axis=-1)
        return vec2mat(A,(self.n_z,self.n_z)), vec2mat(b,(self.n_z,1)), vec2mat(sigma,(self.n_z,self.n_u))

    def initialize(self, sess, z_ref, A_init, b_init, sigma_init, minibatchsize=500, training_epochs=5000, display_step=100):
        n_data = z_ref.shape[0]
        total_batch = int(n_data / minibatchsize)

        for epoch in range(training_epochs):
            avg_loss = 0
            nperm = np.random.permutation(n_data)

            # Loop over all batches
            for i in range(total_batch):
                minibatch_idx = nperm[i * minibatchsize:(i + 1) * minibatchsize]
                batch_zs = z_ref[minibatch_idx, :]
                
                opt, loss = sess.run((self.optimizer, self.loss), feed_dict={self.z_in: batch_zs, self.A_init: A_init, self.b_init:b_init, self.sigma_init: sigma_init})
                avg_loss += loss/total_batch

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss))
                
        print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss))

class GenNet(object):
    """ Neural-net for the generative model: x=g(z) """
    def __init__(self, intermediate_units, n_z):
        self.intermediate_units = intermediate_units
        self.Layers = []

        # Construct the neural network
        for i, unit in enumerate(self.intermediate_units[:-1]):
            self.Layers.append(tf.layers.Dense(units=unit, activation=tf.nn.tanh,
                                               name='GenLayer' + str(i)))  # fully-connected layer w/ relu
        self.Layers.append(tf.layers.Dense(units=self.intermediate_units[-1], name='GenLayerLast'))  # last layer doesn't have relu activation

        # Below is for the later use of this network
        self.z_in = tf.placeholder(tf.float32, (None, n_z))
        x_mean, x_logSig = self.compute_x(self.z_in)
        self.x_out = x_mean
        
    def compute_x(self, z):
        # Input : z=(...,n_z)
        # Output: x=(...,n_x)

        for layer in self.Layers:
            z = layer(z)
        
        x, logSig = tf.split(z,2,axis=-1)
        
        return x, logSig
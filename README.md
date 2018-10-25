# 18-NIPS-APIAE

This is a tensorlfow implementation for the demonstration of Pendulum experiment of the paper: "Adaptive Path-Integral Autoencoder: Representation Learning and Planning for Dynamical Systems":
https://arxiv.org/abs/1807.02128.

## Requirements

- Python 3
- Tensorflow 3
- Numpy
- Pickle

## Instructions

1. To train APIAE for pendulum example, run 'train_pendulum.ipynb'.
The codes were written in ipython with Jupyter Notebook for the better visualization.
This code will save the learned model inside the folder 'weights'.

2. To run pendulum planning demo by using trained APIAE, run 'test_pendulum.ipynb'.
This code will test the reconstruction / prediction / planning performance of APIAE.
Without any modification, this code load the hyperparatmers (i.e. weights of neural networks) from './weights/pendulum_weights_demo.pkl'.
But you may change the name of file if you want to test with new APIAE network trained from step 1.

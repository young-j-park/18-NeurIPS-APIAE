# 18-NIPS-APIAE

This is a tensorlfow implementation for the demonstration of Pendulum experiment of the paper: "Adaptive Path-Integral Autoencoder: Representation Learning and Planning for Dynamical Systems":
https://arxiv.org/abs/1807.02128.

Supplementary video can be found in:
https://www.youtube.com/watch?v=xCp35crUoLQ

## Requirements

- Python 3
- Tensorflow 3
- Numpy
- Pickle

## Instructions

1. To train APIAE for pendulum example, run 'train_pendulum.ipynb'.
The codes were written in ipython with Jupyter Notebook for the better visualization.
This code will save the learned model inside the folder 'weights'.

2. To test the reconstruction / prediction / planning performance of APIAE., run 'test_pendulum.ipynb'.
As a default, this code load the learned model from './weights/pendulum_weights_demo.pkl'.
But you may change the name of the file, if you want to test with new APIAE network trained from step 1.

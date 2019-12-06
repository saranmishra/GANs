"""
This is a work in progress. It is not complete.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import requests 
from data import Stock_Data
import LSTM_Keras
import forecasting
import generator
import discriminator
#from torch import nn, optim
#from torch.autograd.variable import Variable
#from torchvision import transforms, datasets


current_dir = os.getcwd()
print(current_dir)



###################################################################
#This is where Generator and Discriminator will be 
###################################################################
#Generator
generator = generator.GeneratorNet

#Discriminator

discriminator = discriminator.DiscriminatorNet


###################################################################
#This is where training of the networks will occur
###################################################################


def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n

#Optimization

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

#Loss Function
loss = nn.BCELoss()

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

#Training
def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

#Training Generator

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

#Testing
num_test_samples = 10
test_noise = noise(num_test_samples)

#GAN training
# Create logger instance
#logger = Logger(model_name='VGAN', data_name='MNIST')
# Total number of epochs to train
#num_epochs = 230
#for epoch in range(num_epochs):
#    for n_batch, (real_batch,_) in enumerate(data_loader):
#        N = real_batch.size(0)
#        # 1. Train Discriminator
#        real_data = Variable(images_to_vectors(real_batch))
#        # Generate fake data and detach 
#        # (so gradients are not calculated for generator)
#        fake_data = generator(noise(N)).detach()
#        # Train D
#        d_error, d_pred_real, d_pred_fake = \
#              train_discriminator(d_optimizer, real_data, fake_data)
#
#        # 2. Train Generator
#        # Generate fake data
#        fake_data = generator(noise(N))
#        # Train G
#        g_error = train_generator(g_optimizer, fake_data)
#        # Log batch error
#        logger.log(d_error, g_error, epoch, n_batch, num_batches)
#        
#        # Display Progress every few batches
#        if (n_batch) % 100 == 0: 
#            test_images = vectors_to_images(generator(test_noise))
#            test_images = test_images.data
#            logger.log_images(
#                test_images, num_test_samples, 
#                epoch, n_batch, num_batches
#            );
#            # Display status Logs
#            logger.display_status(
#                epoch, num_epochs, n_batch, num_batches,
#                d_error, g_error, d_pred_real, d_pred_fake
#            )




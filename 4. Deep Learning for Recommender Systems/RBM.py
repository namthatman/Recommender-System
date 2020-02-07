# -*- coding: utf-8 -*-
"""
@author: Frank

@modified: nam
"""

import numpy as np
import tensorflow as tf

class RBM(object):

    def __init__(self, visibleDimensions, epochs=20, hiddenDimensions=50, ratingValues=10, learningRate=0.001, batchSize=100):

        self.visibleDimensions = visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.ratingValues = ratingValues
        self.learningRate = learningRate
        self.batchSize = batchSize
        
                
    def Train(self, X):

        for epoch in range(self.epochs):
            np.random.shuffle(X)
            
            trX = np.array(X)
            for i in range(0, trX.shape[0], self.batchSize):
                epochX = trX[i:i+self.batchSize]
                self.MakeGraph(epochX)

            print("Trained epoch ", epoch)


    def GetRecommendations(self, inputUser):
        
        feed = self.MakeHidden(inputUser)
        rec = self.MakeVisible(feed)
        return rec[0]       

    def MakeGraph(self, inputUser):

        # Initialize weights randomly
        maxWeight = -4.0 * np.sqrt(6.0 / (self.hiddenDimensions + self.visibleDimensions))
        self.weights = tf.Variable(tf.random.uniform([self.visibleDimensions, self.hiddenDimensions], minval=-maxWeight, maxval=maxWeight), tf.float32, name="weights")
        
        self.hiddenBias = tf.Variable(tf.zeros([self.hiddenDimensions], tf.float32, name="hiddenBias"))
        self.visibleBias = tf.Variable(tf.zeros([self.visibleDimensions], tf.float32, name="visibleBias"))
        
        # Perform Gibbs Sampling for Contrastive Divergence, per the paper we assume k=1 instead of iterating over the 
        # forward pass multiple times since it seems to work just fine
        
        # Forward pass
        # Sample hidden layer given visible...
        # Get tensor of hidden probabilities
        hProb0 = tf.nn.sigmoid(tf.matmul(inputUser, self.weights) + self.hiddenBias)
        # Sample from all of the distributions
        hSample = tf.nn.relu(tf.sign(hProb0 - tf.random.uniform(tf.shape(hProb0))))
        # Stitch it together
        forward = tf.matmul(tf.transpose(inputUser), hSample)
        
        # Backward pass
        # Reconstruct visible layer given hidden layer sample
        v = tf.matmul(hSample, tf.transpose(self.weights)) + self.visibleBias
        
        # Build up our mask for missing ratings
        vMask = tf.sign(inputUser) # Make sure everything is 0 or 1
        vMask3D = tf.reshape(vMask, [tf.shape(v)[0], -1, self.ratingValues]) # Reshape into arrays of individual ratings
        vMask3D = tf.reduce_max(vMask3D, axis=[2], keepdims=True) # Use reduce_max to either give us 1 for ratings that exist, and 0 for missing ratings
        
        # Extract rating vectors for each individual set of 10 rating binary values
        v = tf.reshape(v, [tf.shape(v)[0], -1, self.ratingValues])
        vProb = tf.nn.softmax(v * vMask3D) # Apply softmax activation function
        vProb = tf.reshape(vProb, [tf.shape(v)[0], -1]) # And shove them back into the flattened state. Reconstruction is done now.
        # Stitch it together to define the backward pass and updated hidden biases
        hProb1 = tf.nn.sigmoid(tf.matmul(vProb, self.weights) + self.hiddenBias)
        backward = tf.matmul(tf.transpose(vProb), hProb1)
    
        # Now define what each epoch will do...
        # Run the forward and backward passes, and update the weights
        weightUpdate = self.weights.assign_add(self.learningRate * (forward - backward))
        # Update hidden bias, minimizing the divergence in the hidden nodes
        hiddenBiasUpdate = self.hiddenBias.assign_add(self.learningRate * tf.reduce_mean(hProb0 - hProb1, 0))
        # Update the visible bias, minimizng divergence in the visible results
        visibleBiasUpdate = self.visibleBias.assign_add(self.learningRate * tf.reduce_mean(inputUser - vProb, 0))

        self.update = [weightUpdate, hiddenBiasUpdate, visibleBiasUpdate]
        
    def MakeHidden(self, inputUser):
        hidden = tf.nn.sigmoid(tf.matmul(inputUser, self.weights) + self.hiddenBias)
        self.MakeGraph(inputUser)
        return hidden
    
    def MakeVisible(self, feed):
        visible = tf.nn.sigmoid(tf.matmul(feed, tf.transpose(self.weights)) + self.visibleBias)
        #self.MakeGraph(feed)
        return visible

# -*- coding: utf-8 -*-
"""
@author: Frank

@modified: nam
"""

import numpy as np
import tensorflow as tf

class AutoRec(object):

    def __init__(self, visibleDimensions, epochs=200, hiddenDimensions=50, learningRate=0.1, batchSize=100):

        self.visibleDimensions = visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.learningRate = learningRate
        self.batchSize = batchSize
        self.optimizer = tf.keras.optimizers.RMSprop(self.learningRate)
        
                
    def Train(self, X):
        
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batchSize):
                epochX = X[i:i+self.batchSize]
                self.run_optimization(epochX)


            print("Trained epoch ", epoch)

    def GetRecommendations(self, inputUser):
        
        # Feed through a single user and return predictions from the output layer.
        rec = self.neural_net(inputUser)
        
        # It is being used as the return type is Eager Tensor.
        return rec[0]

    
    def neural_net(self, inputUser):

        #tf.set_random_seed(0)
        
        # Create varaibles for weights for the encoding (visible->hidden) and decoding (hidden->output) stages, randomly initialized
        self.weights = {
            'h1': tf.Variable(tf.random.normal([self.visibleDimensions, self.hiddenDimensions])),
            'out': tf.Variable(tf.random.normal([self.hiddenDimensions, self.visibleDimensions]))
            }
        
        # Create biases
        self.biases = {
            'b1': tf.Variable(tf.random.normal([self.hiddenDimensions])),
            'out': tf.Variable(tf.random.normal([self.visibleDimensions]))
            }
        
        # Create the input layer
        self.inputLayer = inputUser
        
        # hidden layer
        hidden = tf.nn.sigmoid(tf.add(tf.matmul(self.inputLayer, self.weights['h1']), self.biases['b1']))
        
        # output layer for our predictions.
        self.outputLayer = tf.nn.sigmoid(tf.add(tf.matmul(hidden, self.weights['out']), self.biases['out']))
        
        return self.outputLayer
    
    def run_optimization(self, inputUser):
        with tf.GradientTape() as g:
            pred = self.neural_net(inputUser)
            loss = tf.keras.losses.MSE(inputUser, pred)
            
        trainable_variables = list(self.weights.values()) + list(self.biases.values())
        
        gradients = g.gradient(loss, trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

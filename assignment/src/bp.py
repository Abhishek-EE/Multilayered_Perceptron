#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py
# Author : Abhishek Ranjan Singh

import numpy as np
from src.activation import sigmoid, sigmoid_prime

def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]

    ### Implement here
    # feedforward
    activations = [x]
    a = [x]
    for i in range(0,num_layers-1):
        a.append(biases[i]+np.dot(weightsT[i],activations[i]))
        activations.append(sigmoid(a[-1]))



    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###

    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).df_wrt_a(activations[-1], y)

    ### Implement here
    G = delta

    for k in range(num_layers-1, 0, -1):

    	nabla_b[k-1] = G
    	nabla_wT[k-1] = np.dot(G, np.transpose(activations[k-1]))
    	G = np.dot(np.transpose(weightsT[k-1]), G)
    	G = np.multiply(G, sigmoid_prime(a[k-1]))
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###

    return (nabla_b, nabla_wT)

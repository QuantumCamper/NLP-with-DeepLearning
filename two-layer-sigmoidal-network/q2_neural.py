#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### forward propagation
    z_h = sigmoid(np.dot(data, W1) + b1)
    y_hat = softmax(np.dot(z_h, W2) + b2)
    cost = []
    for i in xrange(len(data)):
        cost.append(np.dot(labels[i],y_hat[i]))
    cost = np.array(cost)

    ### backward propagation

    delta = y_hat - labels
    gradW1 = []
    gradb1 = []
    gradW2 = []
    gradb2 = []
    for i in xrange(len(data)):

         grad_w2 = np.dot(z_h[i].reshape((-1, 1)), delta[i].reshape(1,-1))
         grad_b2 = delta[i].reshape((1,-1))
         delta_z = np.dot(delta[i], W2.transpose())
         delta_a = delta_z * sigmoid_grad(np.dot(data[i], W1) +b1) 
         grad_w1 = np.dot(data[i].reshape((-1,1)), delta_a)
         grad_b1 = delta_a

         print grad_w2.shape
         print grad_b2.shape
         print grad_w1.shape
         print grad_b1.shape

         gradW2.append(grad_w2)
         gradb2.append(grad_b2)
         gradW1.append(grad_w1)
         gradb1.append(grad_b1)
    
    gradW2 = np.array(gradW2)
    gradb2 = np.array(gradb2)
    gradW1 = np.array(gradW1)
    gradb1 = np.array(gradb1)
    print gradW2.shape
    print gradb2.shape
    print gradW1.shape
    print gradb1.shape

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
         gradW2.flatten(), gradb2.flatten()))
    print grad.shape 
    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "check begain"

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()

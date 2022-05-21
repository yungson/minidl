import numpy as np
from .initializer import Initializer
from .math import sigmoid, sigmoid_dt

class Layer(Initializer):

    def __init__(self, input_size, output_size, activation=sigmoid,activation_dt=sigmoid_dt, keep_prob=1):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.activation_dt = activation_dt
        self.keep_prob = keep_prob
        self.weight = self.initialize((output_size, input_size), "he")
        self.bias = self.initialize((output_size, 1), "zeros")
        # moving average of past gradients
        self.weight_v = self.initialize((output_size, input_size), "zeros")
        self.bias_v = self.initialize((output_size, 1), "zeros")
        # moving average of squared past gradients
        self.weight_s = self.initialize((output_size, input_size), "zeros")
        self.bias_s = self.initialize((output_size, 1), "zeros")

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.weight, inputs) + self.bias
        self.a = self.activation(self.z)
        self.m = (np.random.rand(*self.a.shape)<=self.keep_prob)
        self.a = self.a/self.keep_prob
        return self.a

    def backward(self, dZ):
        self.dZ = dZ
        self.dW = np.dot(dZ, self.inputs.T)/self.inputs.shape[1]
        self.db = np.sum(dZ, axis=1, keepdims=True)
        return np.dot(self.weight.T, dZ)

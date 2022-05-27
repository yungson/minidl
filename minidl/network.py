import numpy as np

class Sequential:
    
    def __init__(self, layers):
        self.layers = layers

    def functional(self):
        raise NotImplementedError

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
        
    def backward(self, grad):
        for layer in self.layers[::-1]:
            grad = layer.backward(grad)

    def apply_grads(self, optimizer):
        for layer in self.layers[::-1]:
            layer.apply_grads(optimizer)
    
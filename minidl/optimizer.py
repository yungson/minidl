import numpy as np


    
class SGD:
    
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def __call__(self, layer, t=None):
        layer.weight = layer.weight - self.lr*layer.dW
        layer.bias = layer.bias - self.lr*layer.db
        return layer

class Adam:
    
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def __call__(self, layer, t):
        layer.weight_v = self.beta1*layer.weight_v+(1-self.beta1)*layer.dW
        layer.bias_v = self.beta1*layer.bias_v+(1-self.beta1)*layer.db
        weight_v_corrected = layer.weight_v/(1-self.beta1**t)
        bias_v_corrected = layer.bias_v/(1-self.beta1**t)
        #correction
        layer.weight_s = self.beta2*layer.weight_s+(1-self.beta2)*(layer.dW**2)
        layer.bias_s = self.beta2*layer.bias_s+(1-self.beta2)*(layer.db**2)
        weight_s_corrected = layer.weight_s/(1-self.beta2**t)
        bias_s_corrected = layer.bias_s/(1-self.beta2**t)
        
        layer.weight -= self.lr*weight_v_corrected/(weight_s_corrected**0.5+self.epsilon)
        layer.bias -= self.lr*bias_v_corrected/(bias_s_corrected**0.5+self.epsilon)
        return layer
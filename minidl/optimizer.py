import numpy as np

class Optimizer:

    def __init__(self, **kwargs):
        pass

    def step(self, grad):
        raise NotImplementedError

class SGD(Optimizer):
    
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def step(self, grad):
        step = -self.lr*grad
        return step

class Adam(Optimizer):

    # optimizer should be designed as not to store anything, it's just an optimization tool
    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def step(self, weight):
        # increasing the step
        # computing v, s
        weight.step += 1
        weight.v = self.beta1*weight.v + (1-self.beta1)*weight.grad
        weight.s = self.beta2*weight.s + (1-self.beta2)*(weight.grad**2)
        #correction
        v_corrected = weight.v/(1-self.beta1**weight.step)
        s_corrected = weight.s/(1-self.beta2**weight.step)

        step = self.lr*v_corrected/(s_corrected**0.5+self.epsilon)
        weight.value -= step
        return weight

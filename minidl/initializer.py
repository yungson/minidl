import numpy as np

class Initializer:
    
    def __init__(self):
        pass
    
    def initialize(self, shape, method):
        return eval("self."+method)(shape).astype(np.float32)
        
    def random(self, shape):
        return 0.01*np.random.randn(*shape)
    
    def zeros(self, shape):
        return np.zeros(shape)
    
    def he(self, shape, dist="uniform"):
        """Delving deep into rectifiers: Surpassing human-level performance
        on ImageNet classification" He, K. et al. (2015) 
        """
        if dist == "uniform":
            bound = np.sqrt(6/(shape[1]))
            return np.random.uniform(low=-bound, high=bound, size=shape)
        if dist == "normal":
            std = np.sqrt(2/shape[1])
            return np.random.normal(loc=0.0, scale=std, size=shape)
            
    def xavier(self, shape, dist="uniform"):
        """
        Understanding the difficulty of training deep feedforward neural networks"
        Glorot, X. & Bengio, Y. (2010)
        """
        if dist == "uniform":
            bound = np.sqrt(6/(shape[0]+shape[1]))
            return np.random.uniform(low=-bound, high=bound, size=shape)
        if dist == "normal":
            std = np.sqrt(2/(shape[0]+shape[1]))
            return np.random.normal(loc=0.0, scale=std, size=shape)
        
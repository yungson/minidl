import numpy as np


class DataGenerator:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.m = X.shape[1]
        
    def __call__(self, batch_size, shuffle=True):
        idx = np.arange(self.m)
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, self.m, batch_size):
            batch = idx[i:i+batch_size]
            yield self.X[:,batch], self.y[:,batch]
            
            

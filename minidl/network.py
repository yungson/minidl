import numpy as np

class NeutralNetwork:
    
    def __init__(self, layers, optimizer):
        self.layers = layers
        self.optimizer = optimizer
        
    def forward(self, inputs):
        for each in self.layers:
            inputs = each.forward(inputs)
        self.preds = inputs
    
    def compute_loss(self, label):
        nll = -np.sum(np.log(self.preds)*label, axis=0)
        loss = np.sum(nll)/label.shape[1]
        return loss
    
    def compute_grad(self, label):
        """Notes:
        if the backward portion of each layer has 1/label.shape[1], 
        then the dZ here shouldn't have label.shape[1], or it will make the 
        backpropagated grads too small to learn anything
        dZ = (self.preds - label)/label.shape[1] was wrong and can not learn 
        anything when batch_size>1, but still work well when batch_size==1
        """
        dZ = (self.preds - label)
        return dZ
    
    def backward(self, dZ):
        for each in self.layers[::-1]:
            dZ = dZ*each.m*each.activation_dt(each.z)/each.keep_prob
            dZ = each.backward(dZ)
    
    def update_weights(self, t):
        for each in self.layers[::-1]:
            each = self.optimizer(each,t)
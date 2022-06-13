import numpy as np


class Loss:

    def __init__(self):
        pass

    def compute(self, preds, label):
        raise NotImplementedError

    def grad(self, preds, label):
        raise NotImplementedError

class CrossEntropyLoss():

    def compute(self, preds, label):
        nll = -np.sum(np.log(preds)*label, axis=0)
        loss = np.sum(nll)/label.shape[1]
        return loss

    def grad(self, preds, label):
        """Notes:
        if the backward portion of each layer has 1/label.shape[1], 
        then the dZ here shouldn't have label.shape[1], or it will make the 
        backpropagated grads too small to learn anything. In this case, 
        dZ = (self.preds - label)/label.shape[1] is wrong and can not learn 
        anything when batch_size>1, but still work well by chance when batch_size==1
        So, either divide by label.shape[1] here or everywhere in each layer.
        """
        return (preds - label)/label.shape[1]

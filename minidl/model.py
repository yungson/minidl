import numpy as np
from .evaluate import predict
            
class Model:
    
    def __init__(self, network):
        self.network = network
        self.t = 1
        pass
    
    def train(self, data_gen, batch_size, max_epoch, x_test=None, y_test=None):
        for i in range(max_epoch):
            for (inputs, label) in data_gen(batch_size, shuffle=False):
                self.network.forward(inputs)
                loss = self.network.compute_loss(label)
                grad = self.network.compute_grad(label)                
                self.network.backward(grad)
                self.network.update_weights(self.t)
                self.t +=1
            if x_test is not None and y_test is not None:
                acc, preds = predict(self, x_test, y_test)
                print("epoch: ", i, "loss: ", loss, "val_accuracy: ", acc)
            else:
                print("epoch: ", i, "loss: ", loss)
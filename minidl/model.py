from math import ceil
import numpy as np
            
class Model:
    
    def __init__(self, network, lossfn, optimizer):
        self.network = network
        self.lossfn = lossfn
        self.optimizer = optimizer
        self.step = 0

    def train_one(self, inputs, label):
        preds = self.network.forward(inputs)
        loss = self.lossfn.compute(preds, label)
        grad = self.lossfn.grad(preds, label)
        self.network.backward(grad)
        self.network.apply_grads(self.optimizer)
        return loss

    def train(self, data_gen, batch_size, max_epoch, print_n = 1, print_type="epoch", x_val=None, y_val=None):
        print("Total steps: ", max_epoch*ceil(data_gen.m/batch_size))
        for e in range(1, max_epoch+1):
            for (inputs, label) in data_gen(batch_size, shuffle=True):
                self.step += 1
                loss = self.train_one(inputs, label)
                if print_type == "step" and self.step%print_n == 0:
                    self.output_info(e, loss, x_val, y_val)
            if print_type == "epoch" and self.step%print_n == 0:
                self.output_info(e, loss, x_val, y_val)

    def output_info(self, e, loss, x_val=None, y_val=None):
        # when you don't shuffle and use exactly the same batch data when you print the loss
        # (ie, print_type="epoch", shuffle=False), then you will see your loss strictly going
        # down(before overfitting). However, when you do shuffle=True or because of print_step(="step")
        # does not intercept the same batch data when it prints loss, the loss will go up and down, flucuating 
        if x_val is not None and y_val is not None:
            acc, preds = self.predict(x_val, y_val)
            print(f"[ Epoch {e:04d}, Step {self.step:08d} ] loss: {loss:.6f}, val_accuracy: {acc:.6f}")
        else:
            print(f"[ Epoch {e:04d}, Step {self.step:08d} ] loss: {loss:.6f}")       
                    
    def save(self, save_path, format=""):
        pass

    def load(self):
        pass

    def predict(self, x_test, y_test=None):
        preds = self.network.forward(x_test.T)
        preds = np.argmax(preds,axis=0)
        if y_test is not None:
            acc = sum(preds==y_test)/len(y_test)
            return acc, preds
        return None, preds
    
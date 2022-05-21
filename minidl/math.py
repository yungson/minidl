import numpy as np

def to_onehot(y, label_num):
    return np.eye(label_num)[y.astype(int)]
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_dt(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def softmax(x):
    exp_x = np.exp(x - x.max())
    sum_x = np.sum(exp_x, axis=0, keepdims=True)
    return exp_x/sum_x

def softmax_dt(x):
    return 1
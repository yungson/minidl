import numpy as np

"""
class Layer(Initializer):
    def __init__(self, input_size, output_size, activation=sigmoid,activation_dt=sigmoid_dt, keep_prob=1):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.activation_dt = activation_dt
        self.keep_prob = keep_prob
        self.weight = self.initialize((output_size, input_size), "he")
        self.bias = self.initialize((output_size, 1), "zeros")
        # moving average of past gradients
        self.weight_v = self.initialize((output_size, input_size), "zeros")
        self.bias_v = self.initialize((output_size, 1), "zeros")
        # moving average of squared past gradients
        self.weight_s = self.initialize((output_size, input_size), "zeros")
        self.bias_s = self.initialize((output_size, 1), "zeros")

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.weight, inputs) + self.bias
        self.a = self.activation(self.z)
        self.m = (np.random.rand(*self.a.shape)<=self.keep_prob)
        self.a = self.a/self.keep_prob
        return self.a

    def backward(self, dZ):
        self.dZ = dZ
        self.dW = np.dot(dZ, self.inputs.T)/self.inputs.shape[1]
        self.db = np.sum(dZ, axis=1, keepdims=True)
        return np.dot(self.weight.T, dZ)

"""
class Weight:

    def __init__(self, **kwargs):

        self.value = kwargs["value"]
        # set the grad manually when necessary, may used for debug
        self.grad = kwargs.get("grad", None)
        # how many steps the weight has been optimized, used by some optimizers like Adam
        self.step = kwargs["step"] 
        # moving average of past gradients, for Adam optimizer useage
        self.v = kwargs["v"] 
        # moving average of squared past gradients, for Adam optimizer useage
        self.s = kwargs["s"]            


class Layer():
    def __init__(self):
        pass

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError

    def apply_grads(self, optimizer):
        pass

    @property
    def paras(self, x):
        return self.paras

    @property
    def info(self, x):
        return self.info

class Dense(Layer):

    def __init__(self, w, b):
        self.w = Weight(value=w, v=0, s=0, step=0)
        self.b = Weight(value=b, v=0, s=0, step=0)

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.w.value, self.inputs) + self.b.value

    def backward(self, grad):
        #导数的大小永远等于原有数组的大小
        self.w.grad = np.dot(grad, self.inputs.T) # 此处到底除以还是不除以/self.inputs.shape[1]
        self.b.grad = np.sum(grad, axis=1, keepdims=True) #此处到底是除以还是不除以m， 如果后面除了前面就不用除了
        return np.dot(self.w.value.T, grad)

    def apply_grads(self, optimizer):
        self.w = optimizer.step(self.w)
        self.b = optimizer.step(self.b)

class Dropout(Layer):

    def __init__(self, keep_prob):
        self.keep_prob = keep_prob

    def forward(self, inputs):
        self.inputs = inputs
        self.mask = (np.random.rand(*self.a.shape)<=self.keep_prob) # creating mask
        return self.inputs*self.mask/self.keep_prob # scale the values that are not shut down by mask

    def backward(self, grad):
        return grad*self.mask/self.keep_prob

class Activation(Layer):

    def __init__(self,):
        self.inputs = None

    def compute(self, z):
        raise NotImplementedError

    def compute_derivative(self, x):
        raise NotImplementedError

    def forward(self, inputs):
        self.inputs = inputs
        return self.compute(inputs)

    def backward(self, grad):
        return grad*self.compute_derivative(self.inputs)

class Sigmoid(Activation):

    def compute(self, x):
        return 1/(1+np.exp(-x))

    def compute_derivative(self, x):
        return self.compute(x)*(1-self.compute(x))

class Softmax(Activation):

    def compute(self, x):
        # 那么为什么要对每一个x减去一个max值呢？从需求上来说，如果x的值没有限制的情况下，
        # 当x线性增长，e指数函数下的x就呈现指数增长，一个较大的x（比如1000）就会导致程序的数值溢出，
        # 导致程序error。所以需求上来说，如果能够将所有的x数值控制在0及0以下，则不会出现这样的情况，
        # 这也是为什么不用min而采用max的原因。
        exp_x = np.exp(x - x.max()) 
        sum_x = np.sum(exp_x, axis=0, keepdims=True)
        return exp_x/sum_x

    def compute_derivative(self, x):
        return 1 ##这个结果很意外
import numpy as np

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
        ### For CNN
        for k,v in kwargs.items():
            setattr(self, k, v)


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
        self.w = Weight(value=w, grad=None, v=0, s=0, step=0)
        self.b = Weight(value=b, grad=None, v=0, s=0, step=0)

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


class Conv2D(Layer):

    def __init__(self, w, b, stride, pad):
        """
        fh: height of the filter, fw: width of the filter, in most cases fh=fw
        w: weight, numpy array of shape=(fh, fw, in_channels, out_channels)
        b: bias, numpy array of shape=(1, 1, 1, out_channels)
        pad: padded arrary will become(height+2*pad, width+2*pad)
        """
        self.weight = Weight(value=w, grad=None, v=0, s=0, step=0, stride=stride, pad=pad)
        self.bias = Weight(value=b, grad=None, v=0, s=0, step=0, stride=stride, pad=pad)

    
    def forward(self, inputs):
        """Inputs are the previous activation output of shape=(?height, ?width, in_channels)
        """
        pass

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

class Relu(Activation):

    def compute(self, x):
        return np.maximum(0,x)
    
    def compute_derivative(self, x):
        return x>0

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

class Tanh(Activation):

    def compute(self, x):
        return np.tanh(x)

    def compute_derivative(self, x):
        return 1 - np.power(np.tanh(x),2)

# minidl
 
A simple light-weight deep learning framework to quickly explore and test ideas. It can be used to tailor to personal needs for research. The framework is implemented using a Python Object Oriented Programming where each the whole deep learning process is decomposed into serveral class and each class accomplish its own work, hold its own data when necessary


## installation

```
git clone git@github.com:yungson/minidl.git
export PYTHONPATH=$PYTHONPATH:<the root directory of the downloaded repo>
```

## Class Design

- `datagen.py`: generate the batch data for the neutral network
- `initializer.py`: weight initialization including zeros, random, xavier, He.
- `layer.py`: a neutral network layer including Dense Layer, Dropout, Activation Layers(Sigmoid, Relu, Softmax, Tanh)
- `loss.py`: Loss computation
- `model`: training, predicting,  
- `network.py`: define a neutral network and do forward and backward propagation
- `optimizer.py`: loss optimization class including SGD, Adam. 
- `utils.py`: some util tools such as to-onehot, download.
## examples 

 **DNN**
See [mnist](./examples/mnist/run_mnist.ipynb). 

A simple network structure can achieve `train_accuracy: 0.9922333333333333, test_accuracy: 0.9763`. This is a good demostration that the framework is working correctly.

## TODO

I will add CNN, RNN and attention in the future 🙂


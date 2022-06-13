
import sys
import numpy as np
import pytest 
sys.path.insert(0,"../")
import minidl as mdl

@pytest.fixture
def get_layer():
    np.random.seed(1)

    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    weight = mdl.layer.Weight(value=W1, grad = dW1, v=0, s=0, step=1)
    bias = mdl.layer.Weight(value=b1, grad = db1, v=0, s=0, step=1)
    return weight, bias

def test_adam(get_layer):
    weight, bias = get_layer
    optimizer = mdl.optimizer.Adam(lr=0.01)
    weight = optimizer.step(weight)
    bias = optimizer.step(bias)
    expect_w = np.array([[1.63178673, -0.61919778, -0.53561312],[-1.08040999, 0.85796626, -2.29409733]],dtype=np.float32)
    expect_b = np.array([[1.75225313],[-0.75376553]], dtype=np.float32)
    assert np.allclose(weight.value, expect_w,  atol=1e-08, equal_nan=True)
    assert np.allclose(bias.value, expect_b,  atol=1e-08, equal_nan=True)
import pytest
import numpy as np
#from numpy_model.ANN import NN

"""
model = NN(728, [128, 64], 10)
m = 64
x = np.random.rand(m, 728)
y = np.floor(np.random.rand(m) * 10).astype(int)
model.step(x,y)
"""

def test_weight_init():
    for i in range(model.num_layers):
        assert model.W[i].shape == (model.layers[i + 1], model.layers[i])
        assert np.max(model.W[i]) <= 1.0 / (model.layers[i]) ** 0.5
        assert np.min(model.W[i]) >= -1.0 / (model.layers[i]) ** 0.5


def test_forward_dims():
    for i in range(model.num_layers):
        assert model.a[i].shape == (m, model.layers[i])


def test_backward_dims():
    for i in range(model.num_layers):
        assert model.da[i].shape == model.a[i].shape
        assert model.dz[i].shape == model.z[i].shape
        assert model.dW[i].shape == model.W[i].shape
        assert model.dB[i].shape == model.B[i].shape

class T:
    def __init__(self,x):
        self.x = x
        print(square(self.x))

    def square(self,x):
        return x**2

    def cube(x):
        return x**3

    dict = {'s':square,
        'c': cube}

ob  =T(2)

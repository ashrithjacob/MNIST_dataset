import pytest
import numpy as np
from numpy_model.ANN import NN

model = NN(728, [128, 64], 10)
m =  64
x = np.random.rand(m,728)
model.forward(x)

def test_weight_init():
    for i in range(model.num_layers):
        assert model.W[i].shape == (model.layers[i+1], model.layers[i])
        assert np.max(model.W[i]) <= 1.0/(model.layers[i])**0.5
        assert np.min(model.W[i]) >= -1.0/(model.layers[i])**0.5

def test_forward_dims():
    for i in range(model.num_layers):
        assert model.a[i].shape == (m,model.layers[i])

def test_backward_dims():
    for i in range(model.num_layers):
        assert model.da[i].shape == model.a[i].shape
        assert model.dz[i].shape == model.z[i].shape
        assert model.dW[i].shape == model.W[i].shape
        assert model.dB[i].shape == model.B[i].shape




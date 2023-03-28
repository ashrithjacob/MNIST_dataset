import pytest
import numpy as np
from NeuralNet import NeuralNet
from Layer import Layer
from Data import Data

layers = [784, 128, 64, 10]
fn = ["relu", "relu", "logsoftmax"]
m = 64


@pytest.fixture
def model_instance():
    model = NeuralNet(layers=layers, lr=0.003, mom=0.9)
    return model


def test_NeuralNet_function(model_instance):
    assert model_instance.fn == fn


def test_NeuralNet_layers_shape(model_instance):
    assert len(model_instance.layers) == 3


def test_NeuralNet_buffer(model_instance):
    assert model_instance.B_buf == [None, None, None]
    assert model_instance.W_buf == [None, None, None]


def test_NeuralNet_layers_dimension(model_instance):
    for count, l in enumerate(model_instance.layers):
        assert l.W_t.shape == (layers[count], layers[count+1])
        assert l.B.shape == (1, layers[count + 1])
        assert l.activation == fn[count]

def test_one_hot(model_instance):
    x=model_instance.one_hot(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert x.shape == (9, 10)
    assert x[1, 2] == 1

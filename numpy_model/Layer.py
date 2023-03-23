import numpy as np
import math


class Layer:
    def __init_(self, in_size, out_size, activation="relu"):
        stdv = 1.0 / math.sqrt(in_size)
        # Initialising weights for layer
        self.W = np.random.uniform(-stdv, stdv, size=(out_size, in_size))
        self.B = np.random.uniform(-stdv, stdv, size=(1, out_size))
        self.activation = activation
        self.fn = {
            "relu": {"normal": self.relu, "der": self.d_relu},
            "softmax": {"normal": self.softmax, "der": self.d_softmax},
            "logsoftmax": {"normal": self.logsoftmax, "der": self.d_logsoftmax},
        }

    def relu(self, x):
        x[x < 0] = 0.0
        return x

    def d_relu(self, x):
        x[x < 0] = 0.0
        x[x > 0] = 1.0
        return x

    def softmax(self, x):
        omega = np.sum(np.exp(x), axis=1, keepdims=True)
        return np.exp(x) / omega

    def d_softmax(self, x):
        return self.softmax(x) * (1.0 - self.softmax(x))

    def logsoftmax(self, x):
        return np.log(self.softmax(x))

    def d_logsoftmax(self, x):
        return 1.0 - self.softmax(x)

    def forward(self, a_in):
        z_out = np.matmul(a_in, np.transpose(self.W)) + self.B
        a_out = self.fn[self.activation]["normal"](z_out)
        return {"a": a_out, "z": z_out}

    def backward(self, a_in, z_in, da, m):
        dz = self.fn[self.activation]["der"](z_in) * da
        dW_t = (1.0 / m) * np.matmul(np.transpose(a_in), dz)
        dB = (1.0 / m) * np.sum(dz, axis=0, keepdims=True)
        da_out = np.matmul(dz, self.W)
        return {"da": da_out, "dW_t": dW_t, "dB": dB}
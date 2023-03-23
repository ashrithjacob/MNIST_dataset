import numpy as np
import math
from .Layer import Layer


class NeuralNet:
    def __init__(self, layers, lr=0.003, mom=0.9):
        self.layers_size = layers
        self.lr = lr
        self.mom = mom
        self.num_layers = len(self.layers_size) - 1
        self.classes = self.layers_size[self.num_layers]
        self.fn = []
        self.layers = []
        self.W_buf = []
        self.B_buf = []
        self._init_layers()
        # assert (self.classes == y.max()+1)

    # initialise layer objects and buffers(each layer is object)
    def _init_layers(self):
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                self.fn.append("logsoftmax")
            else:
                self.fn.append("relu")
            l = Layer(
                self.layers_size[i], self.layers_size[i + 1], activation=self.fn[i]
            )
            self.layers.append(l)
            self.B_buf.append(None)
            self.W_buf.append(None)

    def run(self, data_obj):
        loss_epoch = 0
        num_batch = 0
        mini_batch = data_obj.get_minibatch()
        while mini_batch["status"]:
            loss_batch = self.step(mini_batch["x"], mini_batch["y"], mini_batch["size"])
            mini_batch = data_obj.get_minibatch()
            loss_epoch += loss_batch
            num_batch += 1
        # loss for full batch
        loss_epoch = loss_epoch/num_batch
        return loss_epoch

    def step(self, x_batch, y_batch, m):  # a_in is x_batch
        a = []
        z = []
        dW_t = []
        dB = []
        a.append(x_batch)
        # forward pass for 1 minibatch
        for count, l in enumerate(self.layers):
            cache = l.forward(a[count])
            a.append(cache["a"])
            z.append(cache["z"])
        # calculating da[3]
            da = (-1.0 / m) * self.one_hot(y_batch)
        # backward pass for 1 minibatch
        for count, l in reversed(enumerate(self.layers)):
            grad = l.backward(a[count + 1], z[count], da, m)
            da = grad["da"]
            dW_t.append(grad["dW_t"])
            dB.append(grad["dB_t"])
        # updating params
        self.sgd(dW_t, dB)
        loss = self.loss_fn(a[-1], y_batch, m)
        return loss

    def sgd(self, dW_t, dB):
        for count, l in enumerate(self.layers):
            if self.W_buf[count] is None:
                self.W_buf[count] = dW_t[count]
            else:
                self.W_buf[count] *= self.mom + dW_t[count]
            if self.B_buf[count] is None:
                self.B_buf[count] = dB[count]
            else:
                self.B_buf[count] *= self.mom + dB[count]
            l.W_t -= self.lr * self.W_buf
            l.B -= self.lr * self.B_buf

    def loss_fn(self, y_pred,y,m):
        temp = y_pred*self.one_hot(y)
        return -1.0*np.sum(temp)/m
        
    def one_hot(self, y):
        temp = np.zeros((y.shape[0],self.classes))
        temp[np.arange(y.size), y] = 1.0
        


import numpy as np
import math


class NeuralNet:
    def __init__(self, layers, lr=0.003, mom=0.9):
        self.layers = layers
        self.lr = lr
        self.mom = mom
        self.num_layers = len(self.layers)-1
        self.classes = self.layers[self.num_layers]

    def step(self, input_vals, predicted_vals):
        m = input_vals.shape[0]
        self.forward(input_vals)  # creating forward
        self.backward(m, predicted_vals)  # finding d_params
        self.sgd()  # updating params

    def sgd(self):
        for i in range(self.num_layers - 1):
            if self.buf_dW_T[i] is None:
                self.buf_dW_T[i] = self.dW_T[i]
            else:
                self.buf_dW_T[i] *= self.mom + self.dW_T[i]
            if self.buf_dB[i] is None:
                self.buf_dB[i] = self.dB[i]
            else:
                self.buf_dB[i] *= self.mom + self.dB[i]

            self.W[i] -= self.lr * self.buf_dW_T[i]
            self.B[i] -= self.lr * self.buf_dB[i]


class Layer:
    def __init_(self,in_size,out_size, in_a, activation='relu'):
        stdv = 1.0 / math.sqrt(in_size)
        #Initialising weights for layer
        self.W = np.random.uniform(-stdv, stdv, size=(out_size, in_size))
        self.B = np.random.uniform(-stdv, stdv, size=(1, out_size))
        self.activation = activation
        self.in_a = in_a
        self.fn ={'relu':{'normal':self.relu,'der':self.d_relu},
                  'softmax':{'normal':self.softmax,'der':self.d_softmax},
                  'logsoftmax':{'normal':self.logsoftmax, 'der':self.d_logsoftmax}}

    def relu(self,x):
        x[x<0] = 0.0
        return x

    def d_relu(self,x):
        x[x<0] = 0.0
        x[x>0] = 1.0
        return x

    def softmax(self,x):
        omega = np.sum(np.exp(x), axis=1,keepdims=True)
        return (np.exp(x)/omega)
    
    def d_softmax(self,x):
        return (self.softmax(x)*(1.0-self.softmax(x)))
    
    def logsoftmax(self,x):
        return np.log(self.softmax(x))
    
    def d_logsoftmax(self,x):
        return (1.0-self.softmax(x))

    def forward(self):
        z_out = np.matmul(self.in_a,np.transpose(self.W)) + self.B
        a_out = self.fn[self.activation]['normal'](z_out)
        return {'a':a_out,
                'z':z_out}

    def backward(self,z_in,da,m):
        dz = self.fn[self.activation]['der'](z_in)*da
        dW_t = (1.0/m)*np.matmul(np.transpose(self.in_a),dz)
        dB = (1.0/m)*np.sum(dz, axis=0, keepdims=True)
        da_out = np.matmul(dz,self.W)
        return {'da': da_out,
                'dW_t':dW_t,
                'dB':dB}




# TODO
# debug
# clean code
# test
# write main
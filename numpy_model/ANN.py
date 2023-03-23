import numpy as np
import math

class NN():
    def __init__(self, input_layer, hidden_layers, output_layer,lr=0.003,mom=0.9):
        self.layers = []
        self.W = []
        self.B = []
        self.a=[]
        self.z=[]
        self.dW=[]
        self.dB=[]
        self.da=[]
        self.dz=[]
        self.buf_dW=[]
        self.buf_dB=[]
        self.lr = lr
        self.mom = mom
        self.classes = 10
        self.layers.append(input_layer)
        for layer in hidden_layers:
            self.layers.append(layer)
        self.layers.append(output_layer)
        self.num_layers = len(self.layers) - 1
        for i in range(self.num_layers):
            self.__init__weight(self.layers[i],self.layers[i+1])

    def __init__weight(self,input_size, output_size):
        stdv = 1. / math.sqrt(input_size)
        W=np.random.uniform(-stdv, stdv, size=(output_size, input_size))
        B=np.random.uniform(-stdv,stdv, size=(1,output_size))
        self.W.append(W)
        self.B.append(B)
        self.dW.append(None)
        self.dB.append(None)
        self.da.append(None)
        self.dz.append(None)
        self.a.append(None)
        self.z.append(None)
        self.buf_dW.append(None)
        self.buf_dB.append(None)

    def step(self, input_vals, predicted_vals):
        m= input_vals.shape[0]
        self.forward(input_vals)# creating forward
        self.backward(m,predicted_vals)# finding d_params
        self.sgd()#updating params

    def forward(self, input_vals):
        self.a = [input_vals] # extra dim for a
        for i in range(self.num_layers-1):
            self.z[i]=np.matmul(self.a[i],np.transpose(self.W[i])) + self.B[i]
            self.a.append(self.activation(self.z[i]))

    def activation(self,z):
        # check if final layer
        if z.shape[1] == self.layers[self.num_layers-1]:
            #logsoftmax
            omega = np.sum(np.exp(z), axis=1, keepdims=True)
            z = np.log(np.exp(z)/omega)
        else:
            #Relu
            z[z<=0] = 0.0
        return z

    def backward(self,m, predicted_vals):
        Y= np.zeros((m,self.classes))
        Y[np.arange(predicted_vals.size), predicted_vals] = 1.0
        self.da.append((-1.0/m)*Y)# extra dimension for da
        for i in range(self.num_layers-1,-1,-1):
            self.dz[i]=[self.d_activation(self.z[i])* self.da[i+1]]
            self.dW[i]=(1.0/m)*np.matmul(np.transpose(self.a),self.dz[i])
            self.dB[i]=(1.0/m)*np.sum(self.dz[i],axis =0, keepdims=True)
            self.da[i]=np.matmul(self.dz[i],self.W[i])

    def d_activation(self,z):
        # check if final layer
        if z.shape[1] == self.layers[self.num_layers-1]:
            #derivitive of logsoftmax
            omega = np.sum(np.exp(z), axis=1, keepdims=True)
            z = 1-(np.exp(z)/omega)
        else:
            #derivitive of Relu
            z[z<0] = 0.0
            z[z>0] = 1.0
        return z


    def sgd(self):
        for i in range(self.num_layers-1):
            if self.buf_dW[i] is None:
                self.buf_dW[i] = self.dW[i]
            else:
                self.buf_dW[i] *= self.mom + self.dW[i]
            if self.buf_dB[i] is None:
                self.buf_dB[i] = self.dB[i]
            else:
                self.buf_dB[i] *= self.mom + self.dB[i] 
            
            self.W[i] -= self.lr*self.buf_dW[i]
            self.B[i] -= self.lr*self.buf_dB[i]

#TODO
#clean code
#test
# write main
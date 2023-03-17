import numpy as np
from mlxtend.data import loadlocal_mnist

class Data():
    def __init__(self, x_path='../DATA/raw/test-images.bin', y_path ='../DATA/raw/test-labels.bin'):
        self.x_path = x_path
        self.y_path = y_path
        self.x, self.y = loadlocal_mnist(images_path=self.x_path, labels_path=self.y_path)


    def load(self):
        print('Dimensions: %s x %s' % (self.x.shape[0], self.x.shape[1]))
        print('\n1st row', self.x[0].shape)
        #TODO
        #normalise data (see how normalise actually works)
        # split train test
        # convert into mini batches
        # 

class NN():
    def __init__(self, input_layer, hidden_layers, output_layer):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layers = output_layers

    def activation_fn(self):
    def model(self):
    #TODO
    #weights
    #hyperparams
        



if __name__=="__main__":
    ob = Data()
    ob.load()
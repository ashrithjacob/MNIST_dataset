# write main to run code
# create data obj (train and test)
# create NeuralNet obj
# call run module which returns loss for one epoch
# use magic methods

import numpy as np
import math
from Data import Data
from NeuralNet import NeuralNet
from Layer import Layer


if __name__ == "__main__":
    test = Data(norm=(0.5, 0.5), test=True, batch_size=64)
    train = Data(norm=(0.5, 0.5), train=True, batch_size=64)
    model = NeuralNet(layers=[784, 128, 64, 10], lr=0.3, mom=0.9)
    epoch = 50
    for e in range(epoch):
        loss = model.run(train)
        print("loss for epoch {} is {}".format(e, loss))
    accuracy = model.predict(test)
    print("accuracy is {}".format(accuracy))    
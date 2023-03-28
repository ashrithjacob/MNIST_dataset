import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt


class Data:
    def __init__(self, norm=(0.0, 1.0), train=False, test=False, batch_size=64):
        dir = "./DATA/MNIST/raw/"
        self.data = {
            "x_test": dir + "t10k-images-idx3-ubyte",
            "y_test": dir + "t10k-labels-idx1-ubyte",
            "x_train": dir + "train-images-idx3-ubyte",
            "y_train": dir + "train-labels-idx1-ubyte",
        }
        self.train = train
        self.test = test
        self.norm = norm
        self.batch_size = batch_size
        self.batch_index = 0
        self.create()

    def create(self):
        if self.train:
            self.x, self.y = loadlocal_mnist(
                images_path=self.data["x_train"], labels_path=self.data["y_train"]
            )
        elif self.test:
            self.x, self.y = loadlocal_mnist(
                images_path=self.data["x_test"], labels_path=self.data["y_test"]
            )
        else:
            raise ValueError(f"object should be either train or test")
        self.normalise()

    def normalise(self):
        # normalise to [0,1]
        self.x = self.x / np.max(self.x)
        self.x = (self.x - self.norm[0]) / self.norm[1]

    def get_minibatch(self):
        status = True
        start = self.batch_index * self.batch_size
        if start + self.batch_size > self.x.shape[0]:
            end = self.x.shape[0]
        else:
            end = start + self.batch_size
        self.batch_index += 1
        if end == self.x.shape[0]:
            status = False
        return {
            "start": start,
            "x": self.x[start:end, :],
            "y": self.y[start:end],
            "size": end - start,
            "status": status
        }  # returning minibatch and size
    
    def reset(self):
        self.batch_index = 0

    def render(self, data):
        img = np.reshape(data, (28, 28))
        plt.figure(num=None, figsize=(8, 6), dpi=80)
        plt.imshow(img, cmap="gray")
        plt.show()


import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt


class Data:
    def __init__(self, norm=(0.0, 1.0), train=False, test=False, batch_size=64):
        self.data = {
            "x_test": "DATA/test-images.idx3-ubyte",
            "y_test": "DATA/test-labels.idx1-ubyte",
            "x_train": "DATA/train-images.idx3-ubyte",
            "y_train": "DATA/train-labels.idx1-ubyte",
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
        start = self.batch_index * self.batch_size
        if start + self.batch_size > self.x.shape[0]:
            end = self.x.shape()[0]
        else:
            end = start + self.batch_size
        return {
            "x": self.x[start:end, :],
            "y": self.y[start:end],
            "size": end - start,
        }  # returning minibatch and size

    def render(self, data):
        img = np.reshape(data, (28, 28))
        plt.figure(num=None, figsize=(8, 6), dpi=80)
        plt.imshow(img, cmap="gray")
        plt.show()


if __name__ == "__main__":
    ob = Data(norm=(0.5, 0.5), test=True, batch_size=64)
    #ob.display(1)
    batch = ob.get_minibatch()
    ob.render(batch["x"][43,:])
    print(batch["y"][43])
    # print('Dimensions: %s x %s' % (ob.x.shape[0], ob.x.shape[1]))
    # print('\n1st row', ob.x[0].shape)
    # ob.display(0)
    # TODO
    # normalise data (see how normalise actually works)
    # split train test
    # convert into mini batches
    #

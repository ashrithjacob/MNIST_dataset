import torch
from torch import nn

x = torch.empty(3,5)
y = torch.empty(3,5)
z = torch.empty(3,5)
nn.init.uniform_(x)
nn.init.uniform_(y)
nn.init.uniform_(z)
print(x,"\n")
print(y,"\n")
print(z,"\n")
print(x.mul_(y).add(z, alpha = 2))

class mod():
    def __init__(self, input_size=784, output_size=10):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = [128, 64]
        self.model = nn.Sequential(nn.Linear(self.input_size, self.hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(self.hidden_sizes[1], self.output_size),
                      nn.LogSoftmax(dim=1))
        #print(model)
        self.init_weight()

    def init_weight(self):
        for i in range(len(self.model)):
            print(f'{i}th value is: {self.model[i]}')
        print(self.model.parameters())

if __name__ == "__main__":
    model = mod()

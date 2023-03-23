import numpy as np
import torch
from torch import nn

"""
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
"""

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

def check_log_softmax(input):
    m = nn.LogSoftmax(dim=1)
    output = m(input)
    e = torch.exp(input)
    calc = torch.log(e/torch.sum(e, axis =1, keepdim=True),)
    print(output, "\n", calc)

def check_loss_nll(input, target):
    c =0
    l = nn.NLLLoss()
    print(l(input, target))
    for i in range(input.size(0)):
        c -= input[i, target[i]]
    c = c/input.size(0)
    print(c)

if __name__ == "__main__":
    #model = mod()
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss()
    # input is of size N x C = 3 x 5
    input = torch.tensor([[0.0, 5.0, 1.0, 0.0, 0.0],
                         [1.0,1.0,0.0,0.0,0.0],
                         [0.0,0.0,0.0,0.0,1.0],
                         [2.0, 1.0, 1.0, 1.0,0.0]], requires_grad=True)
    # each element in target has to have 0 <= value < C
    target = torch.tensor([1, 0, 4, 3])
    output = loss(m(input), target)
    check_log_softmax(input)
    check_loss_nll(input,target)
    """
    print("m input",m(input))
    print("input",input)
    print("output",output)
    print(output.backward())
    """

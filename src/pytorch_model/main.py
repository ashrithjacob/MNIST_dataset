# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Feb 27 13:04:39 2022

@author: ashrith
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


"""
Defining a function
"""
def view_classify(img, ps):
    """Function for viewing an image and it's predicted classes."""
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis("off")
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title("Class Probability")
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


""""   
#1. Downloading the dataset
"""
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
trainset = datasets.MNIST(
    "./DATA",
    download=True,
    train=True,
    transform=transform,
)
valset = datasets.MNIST(
    "./DATA",
    download=True,
    train=False,
    transform=transform,
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# shuffles the data around


"""
#2. Knowing/undrstanding the datatset
"""
dataiter = iter(trainloader)
images, labels = next(dataiter)

print(images.shape)
print(labels.shape)
# images is tensor of shape [64, 1, 28, 28]
# labels is 64 length tensor containing the labels

plt.imshow(images[0].numpy().squeeze(), cmap="gray_r")

figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis("off")
    plt.imshow(images[index].numpy().squeeze(), cmap="gray_r")


"""
# 3. Building Neural network:
"""
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(
    nn.Linear(input_size, hidden_sizes[0]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    nn.ReLU(),
    nn.Linear(hidden_sizes[1], output_size),
    nn.LogSoftmax(dim=1),
)
print(model)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))

images = images.view(images.shape[0], -1)

logps = model(images)  # log probabilities
loss = criterion(logps, labels)  # calculate the NLL loss


"""
#4. Adjusting weights:
"""
print("Before backward pass: \n", model[0].weight.grad)
loss.backward()
print("After backward pass: \n", model[0].weight.grad)


"""
#5. Training process:
"""
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 10
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        # Training pass
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        # This is where the model learns by backpropagating
        loss.backward()
        # And optimizes its weights here
        optimizer.step()
        running_loss += loss.item()
    print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
print("\nTraining Time (in minutes) =", (time() - time0) / 60)


"""
#6. Testing and evaluation:
"""
images, labels = next(iter(valloader))
img = images[0].view(1, 784)
with torch.no_grad():
    logps = model(img)
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 28, 28), ps)
correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if true_label == pred_label:
            correct_count += 1
        all_count += 1
print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count / all_count))


"""
#7. Save model:
"""
torch.save(model, "./my_mnist_model.pt")

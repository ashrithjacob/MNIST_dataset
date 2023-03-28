# MNIST_dataset_project

Digit recognition projects on the MNIST dataset
The MNIST dataset contains numbers from 0-9 in freehand writing.
The total dataset contains 70000 images with each image being a 28x28 pixel image
We shall explore some simple neural nets through this dataset and implement our custom made neural nets for the same.

Input Data-
Unlike RGB images which have 3 channels, MNIST has only 1 channel, a greyscale channel.

Training Data-
60000 images

Test Data-
10000 images

Evaluation Metric-
Accuracy: The number of correct predictions made by the model divided by the total number of predictions made on the test data

## Network Architecture:
In this project we use 2 hidden layers and an output softmax layer containing 10 nodes.
We use an input layer containing a single channel from the greyscale image(0-255)
here's a heuristic of the Network used:
![Digit_rec](https://github.com/ashrithjacob/MNIST_dataset/blob/master/images/Digit_rec.png?raw=true)

For the loss function, we use a negative likelihood loss function and for the gradient descent a SGD is used.

## Forward and Backward Propogation:
Here are the forward propogation formulae for the layer 'l':\
$z_{l}$ = $a_{l-1}$ x $W^{T}$ + $B$ \
$a_{l}$ = $g_{n}$($z_{l}$)

Where: \
$g_{n}$: activation function for layer $n$
dim($z_{l}$) = dim($a_{l}$) = m x ($length_{output}$) \
dim($W^{T}$) = $length_{output}$ x $length_{input}$ \
dim($B$) = $length_{output}$

Backward propogation formulae: \
$da_{l+1}$ = $dz_{l+1}$ x $W_{l+1}$ \
$dz_{l}$ = 1/m * $g^{'}$($z_{l}$) * $da_{l+1}$ \
$(dw_{l})^{T}$ = 1/m*($(a_{l})^{T}$ x $dz_{l}$) \
$dB_{l}$ = 1/m*sum($dz_{l}$, axis = 0, keepdims = True)
Where: ' * ' is element wise multiplication and ' x ' is matrix multiplication.\

In the backward pass $dz_{3}$ is calculated using the softmax loss function and the derivative of the softmax function, which is: softmax($z_{3}$) - $y_{true}$

## Implementing in Python:
Create a virtual environment and install the requirements:
```
pip install -r requirements.txt
```
For pytorch implementation, run:
```
python3 src/pytorch_model/main.py
```

For numpy implementation (only uses numpy library to build the entire ANN), run:
```
python3 src/numpy_model/main.py
```
## Comparing the two implementations:
The pytorch implementation is much faster than the numpy implementation, as expected:
|           |       Pytorch     |   Numpy   |
-----------------|---------------|-----------|
|   epochs       |       10      |  50       |
|   time        |       2.3 min      |  2.2 min     |
|   accuracy    |       96.7%       |  94.0%      |


## Implementing in C++
run cmake in `c++/build/` directory:
```
cmake ..
```
then run:
```
make
```
[This implementation seems clunky where we need a cmake file followed by make,see:https://github.com/tsoding/nobuild]

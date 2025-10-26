# `model.py` Documentation

This script defines the architecture of the Convolutional Neural Network (CNN) used for MNIST digit classification.

## Class

### `Net()`

This class defines the neural network model by inheriting from `torch.nn.Module`.

#### Architecture

The network is a sequence of layers, each performing a specific mathematical transformation on its input.

1.  **`conv1` & `conv2`: Convolutional Layers**

    A convolutional layer applies a set of learnable filters (kernels) to an input image. This operation is a discrete convolution. For a 2D input image $I$ and a kernel $K$, the output feature map $O$ is given by:

    $$ O(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m, n) + b $$

    Where:
    - `*` denotes the convolution operation.
    - `b` is a learnable bias term.

    In this model:
    - `conv1` takes a 1-channel (grayscale) image and produces 32 feature maps using 3x3 kernels.
    - `conv2` takes the 32 feature maps and produces 64 new feature maps, also with 3x3 kernels.

2.  **ReLU Activation Function**

    After each convolution, a Rectified Linear Unit (ReLU) activation function is applied element-wise. It introduces non-linearity into the model, which is crucial for learning complex patterns.

    $$ \text{ReLU}(x) = \max(0, x) $$

    ![ReLU Plot](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Rectifier_and_softplus_functions.svg/320px-Rectifier_and_softplus_functions.svg.png)

3.  **Max Pooling**

    Max pooling is a down-sampling operation. It reduces the spatial dimensions of the feature maps, which reduces the number of parameters and computation in the network. It also helps to make the representation more robust to small translations of the input.

    A 2x2 max pooling operation takes the maximum value from each 2x2 block of the input feature map.

4.  **`dropout1` & `dropout2`: Dropout**

    Dropout is a regularization technique to prevent overfitting. During training, it randomly sets a fraction of the input units to 0 with a probability $p$ (here $p=0.25$ and $p=0.5$). This can be seen as training a large number of thinned networks.

    The output of a neuron $y$ is multiplied by a random variable $d \sim \text{Bernoulli}(1-p)$.

    During evaluation (inference), dropout is disabled, and the outputs are scaled by a factor of $(1-p)$ to account for the fact that more units are active than during training.

5.  **`fc1` & `fc2`: Fully Connected Layers**

    A fully connected layer applies a linear transformation to the input vector $x$:

    $$ y = Wx + b $$

    Where:
    - $W$ is the weight matrix.
    - $b$ is the bias vector.
    - `fc1` maps the flattened output of the convolutional layers to a 128-dimensional vector.
    - `fc2` maps the 128-dimensional vector to a 10-dimensional vector, corresponding to the 10 digit classes.

6.  **Log Softmax Output**

    The final layer is a `log_softmax` function. This is composed of two functions: softmax and logarithm.

    The **softmax** function converts a vector of $K$ real numbers into a probability distribution of $K$ possible outcomes. For a vector $z = (z_1, ..., z_K)$, the softmax is:

    $$ \text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} $$

    The output of the softmax is a vector of probabilities that sum to 1. The **log_softmax** is simply the logarithm of the softmax output. This is used in conjunction with the Negative Log Likelihood Loss (`NLLLoss`) for training. The combination is mathematically equivalent to `CrossEntropyLoss`, but can be more numerically stable.

#### `forward(self, x)`

This method defines the forward pass of the network, which is the sequence of transformations applied to the input data `x` to produce the output.

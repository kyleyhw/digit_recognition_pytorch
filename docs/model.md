# `model.py` Documentation

This script defines the architecture of the Convolutional Neural Network (CNN) used for MNIST digit classification.

## Class

### `Net()`

This class defines the neural network model by inheriting from `torch.nn.Module`.

#### Architecture

The network consists of the following layers:

1.  **`conv1`**: A 2D convolutional layer. This layer applies a set of learnable filters to the input image. The convolution operation for a single filter is defined as:

    ```
    output(i, j) = sum_{m=0}^{k-1} sum_{n=0}^{k-1} input(i+m, j+n) * kernel(m, n)
    ```

    where `k` is the kernel size. This model uses a 3x3 kernel.

2.  **`conv2`**: Another 2D convolutional layer.

3.  **`dropout1`**, **`dropout2`**: Dropout layers that randomly set a fraction of input units to 0 at each update during training time, which helps prevent overfitting.

4.  **`fc1`**, **`fc2`**: Fully connected (linear) layers that apply a linear transformation to the incoming data: `y = xA^T + b`.

#### `forward(self, x)`

This method defines the forward pass of the network.

##### Description

The forward pass applies the following operations in sequence:

1.  **Convolution and ReLU**: The input `x` is passed through the convolutional layers, and the Rectified Linear Unit (ReLU) activation function is applied. ReLU is defined as:

    ```
    ReLU(x) = max(0, x)
    ```

2.  **Max Pooling**: A 2D max pooling operation is applied. This reduces the spatial dimensions of the output from the convolutional layers. For a 2x2 window, it takes the maximum value in each 2x2 block of the input tensor.

3.  **Flatten**: The 2D feature maps are flattened into a 1D vector to be fed into the fully connected layers.

4.  **Fully Connected and ReLU**: The flattened vector is passed through the fully connected layers, with a ReLU activation in between.

5.  **Log Softmax**: The final output is computed using the `log_softmax` function. This is equivalent to taking the logarithm of the softmax function. The softmax function is defined as:

    ```
    Softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
    ```

    The `log_softmax` is numerically more stable than `log(softmax(x))`.
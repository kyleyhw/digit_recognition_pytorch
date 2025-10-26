# `model.py` Documentation

This script defines the architecture of the Convolutional Neural Network (CNN) used for MNIST digit classification.

## Class

### `Net()`

This class defines the neural network model by inheriting from `torch.nn.Module`.

#### Architecture and Design Rationale

The network is a sequence of layers, each performing a specific mathematical transformation on its input. The design choices for each layer are based on common practices in CNNs for image classification, aiming for a balance between model complexity, performance, and computational efficiency.

1.  **`conv1` & `conv2`: Convolutional Layers**

    A convolutional layer applies a set of learnable filters (kernels) to an input image. This operation is a discrete convolution. For a 2D input image $I$ and a kernel $K$, the output feature map $O$ is given by:

    $$ O(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m, n) + b $$

    Where:
    - `*` denotes the convolution operation.
    - `b` is a learnable bias term.

    **Design Rationale:**
    -   **Two Layers**: A common practice for initial CNN architectures, allowing the model to learn hierarchical features. `conv1` learns basic features (edges, corners), while `conv2` learns more complex patterns from `conv1`'s output.
    -   **3x3 Kernels**: Small kernel sizes are preferred in modern CNNs. They are computationally less expensive than larger kernels and allow for deeper networks. Multiple small kernels can achieve the same receptive field as one large kernel but with more non-linearities and fewer parameters.
    -   **Increasing Filters (32, 64)**: The number of filters (output channels) typically increases with depth in a CNN. This allows the network to learn a richer and more diverse set of features as it processes the input. `conv1` produces 32 feature maps, and `conv2` produces 64.

2.  **ReLU Activation Function**

    After each convolution, a Rectified Linear Unit (ReLU) activation function is applied element-wise. It introduces non-linearity into the model, which is crucial for learning complex patterns.

    $$ \text{ReLU}(x) = \max(0, x) $$

    ![ReLU Plot](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Rectifier_and_softplus_functions.svg/320px-Rectifier_and_softplus_functions.svg.png)

    **Design Rationale:**
    -   **Non-linearity**: Without non-linear activation functions, a neural network would simply be a series of linear transformations, limiting its ability to learn complex, non-linear relationships in data.
    -   **Computational Efficiency**: ReLU is computationally very efficient as it only involves a simple thresholding operation.
    -   **Mitigates Vanishing Gradients**: Unlike sigmoid or tanh functions, ReLU does not suffer from vanishing gradients for positive inputs, which helps in training deeper networks.

3.  **Max Pooling**

    Max pooling is a down-sampling operation. It reduces the spatial dimensions of the feature maps, which reduces the number of parameters and computation in the network. It also helps to make the representation more robust to small translations of the input.

    A 2x2 max pooling operation takes the maximum value from each 2x2 block of the input feature map.

    **Design Rationale:**
    -   **Dimensionality Reduction**: Reduces the spatial size of the representation, which decreases the number of parameters and computational cost.
    -   **Translational Invariance**: Makes the detected features more robust to small shifts or distortions in the input image. The exact position of a feature becomes less important, only its presence within a region.

4.  **`dropout1` & `dropout2`: Dropout**

    Dropout is a regularization technique to prevent overfitting. During training, it randomly sets a fraction of the input units to 0 with a probability $p$ (here $p=0.25$ and $p=0.5$). This can be seen as training a large number of thinned networks.

    The output of a neuron $y$ is multiplied by a random variable $d \sim \text{Bernoulli}(1-p)$.

    During evaluation (inference), dropout is disabled, and the outputs are scaled by a factor of $(1-p)$ to account for the fact that more units are active than during training.

    **Design Rationale:**
    -   **Regularization**: Prevents the network from relying too heavily on any single feature or set of features, forcing it to learn more robust representations. This is crucial for improving generalization to unseen data.
    -   **Dropout Rates (0.25, 0.5)**: These are common values. A higher rate (0.5) is often applied to fully connected layers where overfitting is more common, while a slightly lower rate (0.25) might be used after convolutional layers.

5.  **`fc1` & `fc2`: Fully Connected Layers**

    A fully connected layer applies a linear transformation to the input vector $x$:

    $$ y = Wx + b $$

    Where:
    - $W$ is the weight matrix.
    - $b$ is the bias vector.
    - `fc1` maps the flattened output of the convolutional layers to a 128-dimensional vector.
    - `fc2` maps the 128-dimensional vector to a 10-dimensional vector, corresponding to the 10 digit classes.

    **Design Rationale:**
    -   **Classification Head**: Fully connected layers are typically used at the end of a CNN to perform the final classification based on the high-level features extracted by the convolutional layers.
    -   **Layer Sizes (9216, 128, 10)**:
        -   **9216**: This number is determined by the output dimensions of the last convolutional layer after pooling and before flattening. It's `(output_width * output_height * num_filters_conv2)`. This is not a design choice but a consequence of previous layers.
        -   **128**: A common choice for a hidden layer size in a simple classifier, providing enough capacity to learn complex relationships without being excessively large.
        -   **10**: Corresponds directly to the 10 possible digit classes (0-9) in the MNIST dataset.

6.  **Log Softmax Output**

    The final layer is a `log_softmax` function. This is composed of two functions: softmax and logarithm.

    The **softmax** function converts a vector of $K$ real numbers into a probability distribution of $K$ possible outcomes. For a vector $z = (z_1, ..., z_K)$, the softmax is:

    $$ \text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} $$

    The output of the softmax is a vector of probabilities that sum to 1. The **log_softmax** is simply the logarithm of the softmax output. This is used in conjunction with the Negative Log Likelihood Loss (`NLLLoss`) for training. The combination is mathematically equivalent to `CrossEntropyLoss`, but can be more numerically stable.

    **Design Rationale:**
    -   **Probability Distribution**: Softmax ensures that the output can be interpreted as probabilities for each class, summing to 1.
    -   **Numerical Stability**: Using `log_softmax` directly with `NLLLoss` is numerically more stable than calculating `softmax` and then `log` separately, especially when dealing with very small probabilities.
    -   **Loss Function Compatibility**: It is specifically designed to work efficiently and effectively with `NLLLoss` for multi-class classification tasks.

#### `forward(self, x)`

This method defines the forward pass of the network, which is the sequence of transformations applied to the input data `x` to produce the output.

## Adapting CNNs for Time-Series Data

While Convolutional Neural Networks are most famously used for image processing (2D data), their core concept of local feature extraction through filters can be effectively adapted to other data types, including **time-series data**.

### 1D Convolutions

For time-series data, instead of 2D convolutions, we use **1D convolutions** (`nn.Conv1d` in PyTorch). The fundamental difference lies in the dimensionality of the input and the filter:

-   **2D Convolution (Images)**: A 2D filter (e.g., 3x3) slides across a 2D input (image), performing dot products to extract spatial features.
-   **1D Convolution (Time-Series)**: A 1D filter (e.g., 3x1 or just 3) slides across a 1D input (a sequence of data points over time), performing dot products to extract temporal features.

Mathematically, for a 1D input sequence $S$ and a 1D kernel $K$, the output $O$ is:

$$ O(t) = (S * K)(t) = \sum_{\tau} S(t-\tau) K(\tau) + b $$

Where:
-   $S$ is the input time series.
-   $K$ is the 1D convolutional kernel.
-   $b$ is the bias term.

### Applications to Time-Series Data

1D CNNs are excellent for identifying patterns and features that occur over specific durations or at certain frequencies within a sequence. Examples include:

-   **Sensor Data**: Analyzing sequences of sensor readings (e.g., from accelerometers, temperature sensors) to detect anomalies or classify activities.
-   **Audio Processing**: Extracting features from raw audio waveforms (e.g., for speech recognition, music genre classification).
-   **Financial Data**: Identifying trends or patterns in stock prices or other financial indicators over time.
-   **Natural Language Processing (NLP)**: Although often handled by Recurrent Neural Networks (RNNs) or Transformers, 1D CNNs can be used to extract local features (n-grams) from sequences of word embeddings.

By changing the convolution operation from 2D to 1D, and adjusting the input data format, the powerful feature learning capabilities of CNNs can be extended to a wide array of sequential data problems.

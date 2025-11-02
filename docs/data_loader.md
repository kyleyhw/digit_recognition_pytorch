# `data_loader.py` Documentation

This script is responsible for downloading, loading, and preparing the MNIST dataset for training and testing the neural network.

## Core Concepts

### Tensors

In the context of PyTorch and machine learning, a **tensor** is a multi-dimensional array, analogous to the tensors used in physics and mathematics to represent quantities that have components in different directions. It is a generalization of scalars (0-dimensional tensor), vectors (1-dimensional tensor), and matrices (2-dimensional tensor) [[1]](#ref-bishop-2006).

- A grayscale image, like those in the MNIST dataset, is represented as a 2D tensor (a matrix) of pixel values.
- A batch of images is represented as a 4D tensor with dimensions `(batch_size, channels, height, width)`. For MNIST, `channels` is 1.

### Data Normalization

Data normalization is a crucial preprocessing step in machine learning. The primary goal is to transform the features to be on a similar scale. This helps with the convergence of the optimization algorithm (like gradient descent) used in training [[1]](#ref-bishop-2006).

#### Rationale for Normalization Values (0.5, 0.5)

The normalization in this script transforms the pixel values of the images from the range `[0, 255]` to `[-1, 1]`. This is achieved in two steps:

1.  `transforms.ToTensor()`: This converts the PIL Image (with values in `[0, 255]`) to a PyTorch FloatTensor (with values in `[0.0, 1.0]`).
2.  `transforms.Normalize((0.5,), (0.5,))`: This normalization transforms the data to have a mean of 0 and a standard deviation of 1, if the original data had a uniform distribution. The formula for normalization is:

    $$
    x' = \frac{x - \mu}{\sigma}
    $$

    Where:
    - $x$ is the input pixel value.
    - $\mu$ is the mean.
    - $\sigma$ is the standard deviation.

    In this case, $\mu = 0.5$ and $\sigma = 0.5$. So, for a pixel value `x` in `[0.0, 1.0]`, the transformation is:

    $$
    x' = \frac{x - 0.5}{0.5} = 2x - 1
    $$

    This maps the input range `[0.0, 1.0]` to `[-1.0, 1.0]`. This specific range is often preferred in neural networks as it can help with gradient flow and prevent issues like vanishing/exploding gradients, especially with activation functions like `tanh` (though ReLU is used here, `[-1, 1]` is still a good practice) [[2]](#ref-goodfellow-2016).

## Functions

### `get_data_loaders(batch_size=64)`

This function handles the entire data loading process.

#### Parameters

- `batch_size` (int, optional): The number of samples per batch. Defaults to 64.

#### Rationale for Batch Size (64)

The choice of batch size involves a trade-off:

-   **Larger Batch Sizes**: Can lead to faster training per epoch due to more efficient computation on GPUs. However, they might converge to sharper minima, which can sometimes generalize less well. They also require more memory.
-   **Smaller Batch Sizes**: Introduce more noise into the gradient updates, which can help escape shallow local minima and potentially lead to better generalization. However, they can be slower per epoch and have less stable gradient estimates.

A batch size of **64** is a common and empirically effective choice that balances these factors [[2]](#ref-goodfellow-2016). It's large enough to provide stable gradient estimates and leverage computational parallelism, yet small enough to introduce sufficient noise for good generalization and manage memory efficiently for typical hardware configurations.
#### Returns

- `tuple`: A tuple containing two `DataLoader` objects:
    - `train_loader`: For the training dataset.
    - `test_loader`: For the test dataset.

#### Description

The function performs the following steps:

1.  **Defines a transform:** It creates a sequence of transformations to be applied to the dataset images.
2.  **Downloads datasets:** It downloads the MNIST training and test datasets.
3.  **Creates `DataLoader`s:** It wraps the datasets in `DataLoader` objects.
    - The `train_loader` shuffles the data to ensure that the model sees the data in a different order in each epoch, which helps the model generalize better.
    - The `test_loader` does not need to shuffle the data.

## References

1.  <span id="ref-bishop-2006">Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. [Amazon](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)</span>
2.  <span id="ref-goodfellow-2016">Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. [PDF](https://www.deeplearningbook.org/contents/book.html)</span>

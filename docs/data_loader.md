# `data_loader.py` Documentation

This script is responsible for downloading, loading, and preparing the MNIST dataset for training and testing the neural network.

## Functions

### `get_data_loaders(batch_size=64)`

This function handles the entire data loading process.

#### Parameters

- `batch_size` (int, optional): The number of samples per batch. Defaults to 64.

#### Returns

- `tuple`: A tuple containing two `DataLoader` objects:
    - `train_loader`: For the training dataset.
    - `test_loader`: For the test dataset.

#### Description

The function performs the following steps:

1.  **Defines a transform:** It creates a sequence of transformations to be applied to the dataset images.
    - `transforms.ToTensor()`: Converts the images to PyTorch tensors.
    - `transforms.Normalize((0.5,), (0.5,))`: Normalizes the tensor images. Each channel of the input tensor is normalized with the given mean and standard deviation.

        The normalization is performed using the following formula for each channel:
        
        ```
        output[channel] = (input[channel] - mean[channel]) / std[channel]
        ```

        In this case, with `mean = 0.5` and `std = 0.5`, the formula becomes:

        ```
        output = (input - 0.5) / 0.5 = 2 * input - 1
        ```

        This scales the image pixel values from the range `[0, 1]` to `[-1, 1]`.

2.  **Downloads datasets:** It downloads the MNIST training and test datasets from the internet if they are not already available in the `~/.pytorch/MNIST_data/` directory.

3.  **Creates `DataLoader`s:** It wraps the datasets in `DataLoader` objects, which provide an iterable over the given dataset.
    - The `train_loader` shuffles the data to ensure that the model sees the data in a different order in each epoch.
    - The `test_loader` does not shuffle the data.

## Usage

When the script is run directly, it demonstrates how to use the `get_data_loaders` function and prints some information about the data loaders.

```python
if __name__ == '__main__':
    train_loader, test_loader = get_data_loaders()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
```

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    """
    Downloads the MNIST dataset and creates data loaders for training and testing.

    Args:
        batch_size (int): The batch size for the data loaders.

    Returns:
        tuple: A tuple containing the training and test data loaders.
    """
    # Transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Download and load the test data
    test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    # Example of how to use the function
    train_loader, test_loader = get_data_loaders()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # Get one batch of training images
    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

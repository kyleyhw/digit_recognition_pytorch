import unittest
import torch
from torch.utils.data import DataLoader
from data_loader import get_data_loaders

class TestDataLoader(unittest.TestCase):

    def test_get_data_loaders_return_type(self):
        train_loader, test_loader = get_data_loaders()
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)

    def test_data_loaders_batch_size(self):
        batch_size = 32
        train_loader, _ = get_data_loaders(batch_size=batch_size)
        images, _ = next(iter(train_loader))
        self.assertEqual(images.shape[0], batch_size)

    def test_data_shape_and_type(self):
        train_loader, _ = get_data_loaders(batch_size=1)
        images, labels = next(iter(train_loader))
        
        # Check shape
        self.assertEqual(images.shape, torch.Size([1, 1, 28, 28]))
        self.assertEqual(labels.shape, torch.Size([1]))
        
        # Check type
        self.assertIsInstance(images, torch.FloatTensor)
        self.assertIsInstance(labels, torch.LongTensor)

    def test_data_normalization(self):
        train_loader, _ = get_data_loaders(batch_size=128)
        images, _ = next(iter(train_loader))
        
        # Check if data is normalized between -1 and 1
        self.assertTrue(torch.all(images >= -1.0) and torch.all(images <= 1.0))

if __name__ == '__main__':
    unittest.main()

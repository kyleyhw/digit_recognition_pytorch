import unittest
import torch
from model import Net

class TestModel(unittest.TestCase):

    def test_model_creation(self):
        model = Net()
        self.assertIsInstance(model, Net)

    def test_forward_pass(self):
        model = Net()
        # Create a random input tensor with the shape of a single MNIST image
        input_tensor = torch.randn(1, 1, 28, 28)
        output = model(input_tensor)
        
        # Check if the output has the correct shape (1 sample, 10 classes)
        self.assertEqual(output.shape, torch.Size([1, 10]))

if __name__ == '__main__':
    unittest.main()

import unittest
import torch
import torch.optim as optim
from model import Net
from data_loader import get_data_loaders
from train import train, evaluate

class TestTrain(unittest.TestCase):

    def setUp(self):
        self.model = Net()
        train_loader, test_loader = get_data_loaders(batch_size=4)
        
        # Create a small subset for faster testing
        train_dataset = torch.utils.data.Subset(train_loader.dataset, range(10))
        test_dataset = torch.utils.data.Subset(test_loader.dataset, range(10))
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
        
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=1.0)

    def test_train_function(self):
        # Test if a single training step runs without errors
        try:
            train(self.model, self.train_loader, self.optimizer, epoch=1, log_interval=1)
        except Exception as e:
            self.fail(f"train() function raised an exception: {e}")

    def test_test_function(self):
        # Test if the test function runs without errors
        try:
            evaluate(self.model, self.test_loader)
        except Exception as e:
            self.fail(f"test() function raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()

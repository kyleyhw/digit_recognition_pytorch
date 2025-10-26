import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Fully Connected Layer 1
        self.fc1 = nn.Linear(9216, 128)
        # Fully Connected Layer 2
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # Flatten the tensor
        x = torch.flatten(x, 1)
        # Fully connected layers with ReLU and dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        # Output layer with log_softmax
        output = F.log_softmax(x, dim=1)
        return output

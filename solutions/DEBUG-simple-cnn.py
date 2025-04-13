'''Debug the following script without running it'''

import torch
import torch.nn as nn
from torchvision import datasets, transforms

# MNIST dataset (images are 1-channel 28x28)
transform = transforms.Compose([
    transforms.ToTensor(),
    # No normalization for simplicity
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Define a simple CNN (based on a CIFAR-10 example)
class CNNForCIFAR10(nn.Module):
    def __init__(self):
        super(CNNForCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1) ## SOLUTION: We expect 1 input channel, not 3 input channels
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 classes (digits 0-9)
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNNForCIFAR10()
# Example forward pass on one batch
for images, labels in train_loader:
    outputs = model(images)  # forward pass
    print(outputs.shape)
    break

''' Debug this code without running it '''

import torch
import torch.nn as nn

class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Assuming input images of size 32x32 -> after conv/pool: 64 * 4 * 4 features
        self.fc = nn.Linear(64 * 8 * 8, 100) ## SOLUTION: If we follow the forward model & calculate, we find that the dimensions expected are [8, 64, 8, 8] not [8, 64, 4, 4]
        self.out = nn.Linear(100, 10)
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        print(x.shape)
        # Flatten features
        x = x.view(x.size(0), -1)  ## SOLUTION: We must ensure we flatten per batch, not across all the batches
        x = nn.functional.relu(self.fc(x))
        x = self.out(x)
        return x

model = DeepCNN()
# Test forward pass with dummy input
input_data = torch.randn(8, 3, 32, 32)
output = model(input_data)
print("Output shape:", output.shape)

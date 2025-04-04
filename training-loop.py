'''
Write a PyTorch training loop without using any high-level training utilities. 
Given a model, an optimizer, and a loss function, loop over mini-batches for several epochs: 
perform a forward pass, compute the loss, call backward() to compute gradients, and then optimizer.step() to update parameters. 
Do not forget to zero the gradients (optimizer.zero_grad()) each iteration. This tests your understanding of the training process.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define your model with the necessary layers.
class YourModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(YourModel, self).__init__()
        # TODO: Define your layers, for example:
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # TODO: Implement the forward pass
        # Example:
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # return x
        pass  # Remove this once you implement the forward method

# Create a dummy dataset (replace with your actual data as needed)
# For example, 100 samples with 10 features each and binary targets.
inputs = torch.randn(100, 10)          
targets = torch.randint(0, 2, (100,))    

dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Instantiate your model, loss function, and optimizer.
model = YourModel(input_size=10, hidden_size=5, output_size=2)  # Adjust parameters as needed
criterion = nn.CrossEntropyLoss()  # Or choose a different loss if necessary
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Or use another optimizer

# Training loop scaffolding
num_epochs = 10  # Set the number of epochs as desired
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in dataloader:
        print(epoch)
        # TODO: Zero out gradients from the previous iteration
        # Example: optimizer.zero_grad()

        # TODO: Perform the forward pass to compute outputs from the model
        # Example: outputs = model(batch_inputs)

        # TODO: Compute the loss using the loss function and the model outputs
        # Example: loss = criterion(outputs, batch_targets)

        # TODO: Perform the backward pass to compute gradients
        # Example: loss.backward()

        # TODO: Update the model parameters using the optimizer
        # Example: optimizer.step()

    # TODO: Optionally, add code here to print or log training statistics for each epoch.
    # Example: print(f"Epoch {epoch+1}/{num_epochs} completed")

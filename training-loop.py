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
        # DONE: Define your layers:
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # DONE: Implement the forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x

# Create a dummy dataset
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
    running_loss = 0.0
    for batch_inputs, batch_targets in dataloader:
        # DONE: Zero out gradients from the previous iteration
        optimizer.zero_grad()

        # DONE: Perform the forward pass to compute outputs from the model
        ouptuts = model.forward(batch_inputs)

        # DONE: Compute the loss using the loss function and the model outputs
        loss = criterion(ouptuts, batch_targets)

        # DONE: Perform the backward pass to compute gradients
        loss.backward()

        # DONE: Update the model parameters using the optimizer
        optimizer.step()

        
        # Adding the running loss
        running_loss += loss.item() * batch_inputs.size(0)

    # DONE: Optionally, add code here to print or log training statistics for each epoch.
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch {epoch}: completed. Loss computed: {epoch_loss}')

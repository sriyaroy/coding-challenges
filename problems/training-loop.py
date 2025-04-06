'''
Write a PyTorch training loop without using any high-level training utilities. 
Given a model, an optimizer, and a loss function, loop over mini-batches for several epochs: 
perform a forward pass, compute the loss, call backward() to compute gradients, and then optimizer.step() to update parameters. 
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# TODO: Define your model with the necessary layers.

# TODO: Create a dummy dataset + load it

# TODO: Instantiate your model, loss function, and optimizer.
 # TODO: Adjust parameters as needed
 # TODO: Choose your loss
 # TODO: Choose your optimizer

# TODO: Write the training loop
# TODO: Set the number of epochs as desired
    
      # TODO: Zero out gradients from the previous iteration
      # TODO: Perform the forward pass to compute outputs from the model
      # TODO: Compute the loss using the loss function and the model outputs
      # TODO: Perform the backward pass to compute gradients
      # TODO: Update the model parameters using the optimizer
      # Adding the running loss

    # TODO: Optionally, add code here to print or log training statistics for each epoch.

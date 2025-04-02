'''Implement the forward pass of Batch Normalization for a single layer from scratch (you may use NumPy for the computations). 
Given a batch of inputs (a 2D array of shape [batch_size, num_features]), compute the batch mean and variance, 
then produce the normalized outputs by subtracting the mean and dividing by the standard deviation (plus a small epsilon for numerical stability). 
After normalization, scale and shift the result using learned parameters gamma (scale) and beta (shift). 

(Bonus: implement the backward pass to compute gradients with respect to the inputs, gamma, and beta.)*'''

import numpy as np
import torch
import torch.nn as nn

class BatchNorm:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        """
        Initialize the BatchNorm layer.

        Args:
            num_features (int): Number of features in the input.
            epsilon (float): Small constant for numerical stability.
            momentum (float): Momentum for running mean and variance.
        """
        # Learnable parameters: scale (gamma) and shift (beta)
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.epsilon = epsilon

        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.momentum = momentum

        # Cache for backward pass (if needed)
        self.cache = None

    def forward(self, x, training=True):
        """
        Forward pass for batch normalization.

        Args:
            x (np.ndarray): Input array of shape [batch_size, num_features].
            training (bool): True if forward pass is for training.

        Returns:
            out (np.ndarray): Batch-normalized output.
        """
        if training:
            # TODO: Compute the batch mean and variance over the mini-batch.
            rows, cols = x.shape
            print(x)
            mean_variance_matrix = np.zeros((2, cols))
            for c in range(cols):
                # Calculate the mean
                mean_variance_matrix[0, c] = np.sum(x[:, c]) / rows
                # Calculate the variance
                mean_variance_matrix[1, c] = np.sum(np.square(x[:, c] - mean_variance_matrix[0, c])) / rows
                
            # TODO: Normalize the input using the computed statistics.
            # TODO: Scale and shift the normalized input using gamma and beta.
            # TODO: Update running_mean and running_var using momentum.
            # Save necessary variables in self.cache for use in the backward pass.
            out = None  # Replace with your computed output.
        else:
            # TODO: Normalize using the running statistics (running_mean and running_var).
            # TODO: Scale and shift the normalized input.
            out = None  # Replace with your computed output.

        return out

    def backward(self, dout):
        """
        (Bonus) Backward pass for batch normalization.

        Args:
            dout (np.ndarray): Upstream gradients of shape [batch_size, num_features].

        Returns:
            dx (np.ndarray): Gradient with respect to input x.
            dgamma (np.ndarray): Gradient with respect to scale parameter gamma.
            dbeta (np.ndarray): Gradient with respect to shift parameter beta.
        """
        # TODO: Use self.cache to compute the gradients.
        dx = None  # Replace with your computed gradient w.r.t. input.
        dgamma = None  # Replace with your computed gradient w.r.t. gamma.
        dbeta = None  # Replace with your computed gradient w.r.t. beta.
        return dx, dgamma, dbeta

# Example usage (for testing purposes, not part of the solution):
if __name__ == "__main__":
    # Generate dummy input data: batch_size x num_features using NumPy
    np.random.seed(0)
    x_np = np.random.randn(10, 5).astype(np.float32)

    # Create our custom BatchNorm layer instance
    bn_custom = BatchNorm(num_features=5)

    # Forward pass using our custom implementation (training mode)
    out_custom = bn_custom.forward(x_np, training=True)
    print("Custom BatchNorm (training):", out_custom)

    # Forward pass using our custom implementation (inference mode)
    out_custom_infer = bn_custom.forward(x_np, training=False)
    print("Custom BatchNorm (inference):", out_custom_infer)

    # ----- Compare with PyTorch's BatchNorm1d -----
    # Set the same random seed for reproducibility.
    torch.manual_seed(0)
    x_torch = torch.from_numpy(x_np)

    # Create a PyTorch BatchNorm1d layer with similar parameters.
    # Note: Ensure that the momentum and epsilon are set similarly.
    bn_torch = nn.BatchNorm1d(num_features=5, eps=1e-5, momentum=0.9)
    bn_torch.train()  # set to training mode

    # Forward pass using PyTorch's implementation
    out_torch_train = bn_torch(x_torch).detach().numpy()
    print("PyTorch BatchNorm (training):", out_torch_train)

    bn_torch.eval()  # switch to inference mode
    out_torch_infer = bn_torch(x_torch).detach().numpy()
    print("PyTorch BatchNorm (inference):", out_torch_infer)

    # Note:
    # Comparing your custom implementation to PyTorch's output is a good sanity check.
    # However, differences in how running averages and momentum are updated might lead
    # to slight discrepancies. Make sure to align the behavior (especially with the update rules)
    # when comparing.

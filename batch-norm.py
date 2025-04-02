'''Implement the forward pass of Batch Normalization for a single layer from scratch (you may use NumPy for the computations). 
Given a batch of inputs (a 2D array of shape [batch_size, num_features]), compute the batch mean and variance, 
then produce the normalized outputs by subtracting the mean and dividing by the standard deviation (plus a small epsilon for numerical stability). 
After normalization, scale and shift the result using learned parameters gamma (scale) and beta (shift). 

(Bonus: implement the backward pass to compute gradients with respect to the inputs, gamma, and beta.)*'''

import numpy as np

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

        # Cache for backward pass
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
    # Generate dummy input data: batch_size x num_features
    x = np.random.randn(10, 5)

    # Create a BatchNorm layer instance
    bn = BatchNorm(num_features=5)

    # Forward pass (training mode)
    out_train = bn.forward(x, training=True)
    print("Forward pass (training):", out_train)

    # Forward pass (inference mode)
    out_infer = bn.forward(x, training=False)
    print("Forward pass (inference):", out_infer)

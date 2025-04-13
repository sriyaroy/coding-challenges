'''
Implement the 2D convolution operation on a grayscale image from scratch using NumPy with no padding, stride or dilation. 
Given an input image (as a 2D array) and a kernel (filter) matrix, write a function to compute the convolution result without using any built-in convolution functions.
'''
import numpy as np

def conv2d(image, kernel):
    """
    Compute 2D convolution on a grayscale image using NumPy.

    Parameters:
        image (np.ndarray): A 2D array representing the grayscale image.
        kernel (np.ndarray): A 2D array representing the convolution kernel.

    Returns:
        np.ndarray: The convolution result as a 2D array.
    """
    # DONE: Prepare an output array of the same shape as the original image.
    # Formula for output of convolution is [(W - K + 2P)/S] + 1
    m, n = image.shape
    km, kn = kernel.shape
    
    om, on = m - km + 1, n - kn + 1
    output = np.empty(((om, on)))

    
    # DONE: Implement the convolution operation: SOLUTION 1
    # - Iterate over each pixel of the padded image (or use a vectorized approach if possible)
    # - Apply the kernel to the appropriate region of the image
    windows = np.lib.stride_tricks.sliding_window_view(image, (km, kn)) ## this is a vectorised approach! However using loops below is not, we could vectorise it if we wanted (see solution 2)
    for p in range(om):
        for w in range(on):
            # - Compute the sum of the element-wise multiplication between the flipped kernel and the image patch
            # - Store the result in the output array.
            output[p, w] = np.sum(windows[p, w] * kernel.T)

    '''# SOLUTION 2: Fully vectorised approach
    output2 = np.tensordot(windows, kernel, axes=([2,3],[0,1]))''' # Here we set axes to the 3rd & 4th in windows as that's what we want to sum over in our 4D array windows
    
    # DONE: Return the output array.
    return output  # Replace this with your actual output after computing the convolution

def main():
    # DONE: Generate or load a sample grayscale image as a 2D numpy array.
    # For example, create a dummy image of size 256x256:
    image = np.random.rand(10, 10)
    print(image)
    
    # DONE: Define a convolution kernel (filter), e.g., a 3x3 matrix.
    kernel = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    
    # DONE: Call your convolution function with the image and kernel.
    result = conv2d(image, kernel)
    
    # DONE: Optionally, add code to visualize or validate the result.
    print("Convolution result:")
    print(result)
    ## If we print the shape of our output, we'll see that it matches with our formula
    print(result.shape)

if __name__ == "__main__":
    main()

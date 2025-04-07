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
    # TODO: Prepare an output array of the same shape as the original image.
    output = 0
    
    # TODO: Implement the convolution operation:
    # - Iterate over each pixel of the padded image (or use a vectorized approach if possible)
    # - Apply the kernel to the appropriate region of the image
    # - Compute the sum of the element-wise multiplication between the kernel and the image patch
    # - Store the result in the output array.
    
    # Make sure your implementation handles the boundaries correctly using the padded image.
    
    # TODO: Return the output array.
    return output  # Replace this with your actual output after computing the convolution

def main():
    # TODO: Generate or load a sample grayscale image as a 2D numpy array.
    # For example, create a dummy image of size 256x256:
    image = np.random.rand(256, 256)
    
    # TODO: Define a convolution kernel (filter), e.g., a 3x3 matrix.
    kernel = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    
    # TODO: Call your convolution function with the image and kernel.
    result = conv2d(image, kernel)
    
    # TODO: Optionally, add code to visualize or validate the result.
    print("Convolution result:")
    print(result)
    
if __name__ == "__main__":
    main()

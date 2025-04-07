'''
Implement the 2D convolution operation on a grayscale image from scratch using NumPy. 
Given an input image (as a 2D array) and a kernel (filter) matrix, write a function to compute the convolution result without using any built-in convolution functions. 
Ensure you handle the edges properly.
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
    # TODO: Determine the necessary padding size based on the kernel dimensions.
    # Hint: Typically, for a kernel of size (k, k), you might use a padding of size k//2.
    
    # TODO: Apply zero-padding to the input image to handle the edges.
    # Example: Use np.pad() to add the required border of zeros.
    
    # TODO: Prepare an output array of the same shape as the original image.
    output = np.empty_like(image)
    
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

    # TODO: Add test for input that is (b, h, w)

if __name__ == "__main__":
    main()

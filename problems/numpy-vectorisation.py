''' 
Take a naive image processing task that is initially implemented with Python loops and rewrite it using NumPy vectorized operations. 
Compare the execution time of the loop version vs. the vectorized version on a large image to demonstrate the performance gain.
'''

import numpy as np
import time

def naive_image_processing(image):
    """
    TODO: Implement a naive image processing operation using Python loops.
          For instance, you might iterate over each pixel to perform a computation.
    """
    result = image
    return result

def vectorized_image_processing(image):
    """
    TODO: Implement the same image processing operation using NumPy vectorized operations.
          Aim to avoid explicit Python loops.
    """
    # TODO: Add your vectorized operation with numpy
    result = image 
    return result

def main():
    # Generate a large dummy image (e.g., a 1024x1024 grayscale image)
    large_image = np.random.rand(1024, 1024)
    
    result_naive = naive_image_processing(large_image)
    result_vectorized = vectorized_image_processing(large_image)

if __name__ == "__main__":
    main()

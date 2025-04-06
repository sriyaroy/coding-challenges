''' 
Take a naive image processing task that is initially implemented with Python loops and rewrite it using NumPy vectorized operations. 
Compare the execution time of the loop version vs. the vectorized version on a large image to demonstrate the performance gain.
'''

import numpy as np
import time

def naive_image_processing(image):
    """
    DONE: Implement a naive image processing operation using Python loops.
          For instance, you might iterate over each pixel to perform a computation.
    """
    # Create a placeholder for the output (adjust shape or type as needed)
    result = np.empty_like(image)
    
    # Standardising every pixel in the image (ie. mean = 0 and variance = 1)
    m, n = len(result), len(result[0])

    # First calculate the mean & variance across each row
    mean = 0
    variance = 0
    for r in range(m):
        mean += sum(image[r]) / len(image[r])

    mean /= m

    ## calculating variance (1/N ∑ (x - mean)^2)
    for r in range(m):
        for c in range(n):
            variance += (image[r][c] - mean)**2

    variance /= m*n

    ## Standardising every pixel
    for r in range(m):
        for c in range(n):
            result[r][c] = (image[r][c] - mean) / (variance ** 0.5)

    print(mean, variance ** 0.5)
    return result

def vectorized_image_processing(image):
    """
    TODO: Implement the same image processing operation using NumPy vectorized operations.
          Aim to avoid explicit Python loops.
    """
    # DONE: Replace the following line with your vectorized operation
    result = (image - image.mean()) / image.std()
    print(image.mean(), image.std())
    return result

def main():
    # Generate a large dummy image (e.g., a 1024x1024 grayscale image)
    large_image = np.random.rand(1024, 1024)
    
    # Time the naive loop-based image processing function
    start_time = time.time()
    result_naive = naive_image_processing(large_image)
    naive_duration = time.time() - start_time
    print(f"Naive implementation time: {naive_duration:.6f} seconds")
    
    # Time the vectorized image processing function
    start_time = time.time()
    result_vectorized = vectorized_image_processing(large_image)
    vectorized_duration = time.time() - start_time
    print(f"Vectorized implementation time: {vectorized_duration:.6f} seconds")

if __name__ == "__main__":
    main()

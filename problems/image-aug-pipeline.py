'''Write a function to perform random image augmentations such as horizontal flips, small rotations (e.g., ±15°), 
and brightness adjustments. The function should take an input image and return a new, randomly transformed image. 
You can use libraries like OpenCV or PIL to apply these transformations. 
Ensure that each augmentation is applied with some probability or random magnitude to simulate how it is done during training. 
This tests familiarity with common augmentation techniques to make models more generalizable.'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

import random

def random_augmentation(image):
    # TODO: randomly select the augmentation to be applied for a given image
    return image

def horizontal_flip(image):
    # TODO: Implement horizontal flip
    return image

def rotate15to45(image):
    # TODO: Random rotations
    return image

def brightness(image):
    # TODO: Implement Brightness Adjustments
    return image


if __name__ == "__main__":
    input_image = cv2.imread('') # Adding your image path
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    augmented_image = random_augmentation(input_image)
    
    # TODO: Display the original image vs the augmented one to view the results

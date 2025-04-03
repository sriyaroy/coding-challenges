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

    random_mag = random.random()
    if random_mag < 0.33:
        image = horizontal_flip(image)
    if 0.33 < random_mag < 0.66:
        image = rotate15to45(image)
    else:
        image = brightness(image)

    return image

def horizontal_flip(image):
    # DONE: Implement horizontal flip
    image = cv2.flip(image, 0)
    return image

def rotate15to45(image):
    # DONE: Random rotations
    # Find the image centre
    centre = (image.shape[1]//2, image.shape[0]//2)
    angle = random.randrange(15, 45)

    # Generate the rotation matrix
    rot_mat = cv2.getRotationMatrix2D(centre, angle, 1)
    image = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))
    return image

def brightness(image):
    # DONE: Implement Brightness Adjustments
    image = cv2.convertScaleAbs(image, alpha=1, beta=random.randrange(-100, 100))
    return image


if __name__ == "__main__":
    input_image = cv2.imread('1200px-Sunflower_from_Silesia2.jpg')
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    augmented_image = random_augmentation(input_image)
    
    # Display or save augmented_image as needed
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image)
    plt.title('Augmented Image')
    plt.show()

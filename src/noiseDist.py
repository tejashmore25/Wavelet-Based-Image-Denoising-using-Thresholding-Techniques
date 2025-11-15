from skimage.util import random_noise
import numpy as np

# Add Noise Function
def addGaussianNoise(image, mean = 0, sigma = 15):        
    var = (sigma / 255) ** 2
    noisy_img = random_noise(image, mode='gaussian', mean = mean, var = var)
    return noisy_img

def add_salt_and_pepper_noise(image, amount = 0.05):      
    noisy_img = random_noise(image, mode='s&p', amount=amount)
    return noisy_img

def add_uniform_noise(image, low = -0.1, high = 0.1):
    m, M = np.min(np.ravel(image)), np.max(np.ravel(image))
    noise = np.random.uniform(low, high, image.shape)
    noisy_img = image + noise
    noisy_img = np.clip(noisy_img, m, M)
    return noisy_img

# Master Noise Function
def addNoise(image, mode = 'gaussian', mean = 0, sigma = 15, amount = 0.05, low = -0.1, high = 0.1):
    if image is not None:
        if mode == 'gaussian':
            return addGaussianNoise(image, mean, sigma)
        elif mode == 's&p':
            return add_salt_and_pepper_noise(image, amount)
        elif mode == 'uniform':
            return add_uniform_noise(image, low, high)
    return image
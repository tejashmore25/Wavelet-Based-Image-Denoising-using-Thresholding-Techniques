import numpy as np
from cv2 import GaussianBlur, medianBlur

#Threshold
def applyThreshold(coeffs, threshold):
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)

# Estimate Noise Standard Deviation from Wavelet Coefficient
def get_sigma_est(coeffs_list):
    """Estimates noise standard deviation from the HH1 sub-band."""
    hh1 = coeffs_list[-1][2] # HH1 band
    median_hh1 = np.median(np.abs(hh1))
    if median_hh1 == 0: 
        return 0.0001 # Avoid division by zero
    sigma_est = median_hh1 / 0.6745
    return sigma_est

#Evaluation
def mse(img1, img2):
  return np.mean((img1 - img2) ** 2)

# Smoothing Function
def gaussianSmoothing(noisy_image, size = 5, sigma = 0):
    return GaussianBlur(noisy_image, (size, size), sigma)

def medianSmoothing(noisy_image, size = 5):
    return medianBlur(noisy_image, size)
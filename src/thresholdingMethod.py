import numpy as np
from .utilities import applyThreshold

def get_visushrink_threshold(N, sigma_est):       
    # Calculate the threshold
    threshold = sigma_est * np.sqrt(2 * np.log(N))
    return threshold

def get_sureshrink_threshold(subband_coeffs, sigma_est):
    if sigma_est == 0: 
        return 0
    n = subband_coeffs.size
    if n == 0: 
        return 0
    
    y = subband_coeffs.ravel() / sigma_est
    y_abs = np.abs(y)
    y_abs_sorted = np.sort(y_abs)
    y_sq = y_abs_sorted**2
    
    term2 = 2 * np.arange(1, n + 1)
    
    cumsum_y_sq = np.cumsum(y_sq)
    term1 = cumsum_y_sq + y_sq * np.arange(n - 1, -1, -1)
    risk_values = n - term2 + term1

    # Find the minimum risk
    best_risk_index = np.argmin(risk_values)
    
    # Hybrid Approach
    T_sure = y_abs_sorted[best_risk_index]
    T_univ = np.sqrt(2 * np.log(n)) 
    eta = np.mean(np.maximum(0, y_sq - 1)**2)
    magic_val = np.log(n)**(1.5) / np.sqrt(n)
    
    if eta < magic_val:
        final_threshold = T_univ
    else:
        final_threshold = T_sure
    return final_threshold * sigma_est

def get_bayesshrink_threshold(subband_coeffs, sigme_est):
    # noise variance
    sigma_n_sq = sigme_est ** 2
    
    n = subband_coeffs.size
    if n == 0: 
        return 0
    
    # subband variance
    sigma_y_sq = np.mean(subband_coeffs**2)
    # Estimating true signal variance
    sigma_w_sq = sigma_y_sq - sigma_n_sq
    
    # Handling if noise variance is greater than sub-band variance
    if sigma_w_sq <= 0:
        return np.inf 
    
    sigma_w = np.sqrt(sigma_w_sq)
    T_bayes = sigma_n_sq / sigma_w
    return T_bayes

# Apply Thresholding Method
def applyShrink(coeffs, sigma_est, N,  mode = 'visu'):
    thresholded_details = [coeffs[0]]
    if mode == 'visu':
        T = get_visushrink_threshold(N, sigma_est)
        for (LH, HL, HH) in coeffs[1:]:
            LH_t = applyThreshold(LH, T)
            HL_t = applyThreshold(HL, T)
            HH_t = applyThreshold(HH, T)
            thresholded_details.append((LH_t, HL_t, HH_t))

    elif mode == 'sure':
        for (LH, HL, HH) in coeffs[1:]:
            # Calculate a *unique* threshold for each band
            T_LH = get_sureshrink_threshold(LH, sigma_est)
            T_HL = get_sureshrink_threshold(HL, sigma_est)
            T_HH = get_sureshrink_threshold(HH, sigma_est)

            # Apply the unique threshold (SUREShrink only uses soft thresholding)
            LH_t = applyThreshold(LH, T_LH)
            HL_t = applyThreshold(HL, T_HL)
            HH_t = applyThreshold(HH, T_HH)
            thresholded_details.append((LH_t, HL_t, HH_t))

    elif mode == 'bayes':
        for (LH, HL, HH) in coeffs[1:]:
            T_LH = get_bayesshrink_threshold(LH, sigma_est)
            T_HL = get_bayesshrink_threshold(HL, sigma_est)
            T_HH = get_bayesshrink_threshold(HH, sigma_est)
            
            #Apply the Threshold
            LH_t = applyThreshold(LH, T_LH)
            HL_t = applyThreshold(HL, T_HL)
            HH_t = applyThreshold(HH, T_HH)
            thresholded_details.append((LH_t, HL_t, HH_t))

    return thresholded_details
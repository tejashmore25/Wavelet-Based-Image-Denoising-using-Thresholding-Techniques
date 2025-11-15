import numpy as np
from .utilities import applyThreshold

def get_visushrink_threshold(N, sigma_est):       
    # Calculate the threshold
    threshold = sigma_est * np.sqrt(2 * np.log(N))
    return threshold

def get_sureshrink_threshold(subband_coeffs, sigma_est):
    """
    Calculates the optimal threshold for a *single* sub-band
    using Stein's Unbiased Risk Estimate (SURE).
    
    Args:
        subband_coeffs: The 2D array of wavelet coefficients for one sub-band.
        sigma: The estimated noise standard deviation (from HH1).
    """
    if sigma_est == 0: return 0 # No noise, no threshold
    
    n = subband_coeffs.size
    if n == 0: return 0
    
    # Rescale coefficients by sigma for SURE calculation (assumes sigma=1)
    y = subband_coeffs.ravel() / sigma_est
    
    # 1. Get the sorted absolute values
    y_abs = np.abs(y)
    y_abs_sorted = np.sort(y_abs)
    y_sq = y_abs_sorted**2
    
    # --- CORRECTED RISK CALCULATION ---
    # This is the correct vectorized formula for SURE(T)
    # risk = n - 2*count(|y|<=T) + sum(min(y^2, T^2))
    
    # For a threshold T = y_abs_sorted[k]:
    
    # term 'count(|y|<=T)' is (k+1)
    term2 = 2 * np.arange(1, n + 1)
    
    # 'cumsum_y_sq[k]' is sum(y_sq[i] for i <= k)
    cumsum_y_sq = np.cumsum(y_sq)
    
    # 'sum(min(y^2, T^2))' is cumsum_y_sq[k] + y_sq[k] * (n - (k+1))
    term1 = cumsum_y_sq + y_sq * np.arange(n - 1, -1, -1)
    
    risk_values = n - term2 + term1
    # --- END CORRECTION ---

    # 3. Find the minimum risk
    best_risk_index = np.argmin(risk_values)
    
    # The threshold that gave this minimum risk
    T_sure = y_abs_sorted[best_risk_index]
    
    # 4. Implement the "Hybrid" rule (this part was correct)
    T_univ = np.sqrt(2 * np.log(n)) # On the rescaled data
    
    eta = np.mean(np.maximum(0, y_sq - 1)**2)
    magic_val = np.log(n)**(1.5) / np.sqrt(n)
    
    if eta < magic_val:
        # Sub-band is sparse, use the safer universal threshold
        final_threshold = T_univ
    else:
        # Sub-band has signal, use the SURE-minimized threshold
        final_threshold = T_sure
        
    # 5. Return the threshold, scaled back up by sigma
    return final_threshold * sigma_est

def get_bayesshrink_threshold(subband_coeffs, sigme_est):
    """
    Calculates the optimal threshold for a *single* sub-band
    using Bayesian analysis.
    
    Args:
        subband_coeffs: The 2D array of wavelet coefficients for one sub-band.
        sigma_n_sq: The *variance* (sigma^2) of the noise, estimated from HH1.
    """
    sigma_n_sq = sigme_est ** 2
    n = subband_coeffs.size
    if n == 0: return 0
    
    # 1. Calculate sigma_y_sq (variance of the noisy sub-band)
    #    We assume mean is ~0 for detail coefficients. Var = E[X^2] - (E[X])^2
    sigma_y_sq = np.mean(subband_coeffs**2)
    
    # 2. Estimate sigma_w_sq = max(0, sigma_y^2 - sigma_n^2)
    sigma_w_sq = sigma_y_sq - sigma_n_sq
    
    # 3. Handle case where noise variance is greater than sub-band variance
    if sigma_w_sq <= 0:
        # This sub-band is considered all noise.
        # Threshold should be max possible value to kill all coeffs.
        return np.inf 
    
    # 4. Calculate threshold T_B = sigma_n^2 / sigma_w
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
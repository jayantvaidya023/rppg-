"""
hrv_analysis.py — HRV extraction from the rPPG pulse signal.

This module is ADDITIVE: it does NOT replace the FFT-based BPM estimation
in heartrate.py / rppg_core.py. Instead, it provides systolic peak detection,
RR interval computation, and standard HRV metrics on top of the existing
filtered POS signal.
"""

import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Systolic peak detection with adaptive thresholds
# ---------------------------------------------------------------------------

def detect_systolic_peaks(signal, fs, min_bpm=42, max_bpm=180):
    """
    Detect systolic peaks in the filtered POS signal.

    Uses scipy.signal.find_peaks with:
        - Adaptive height threshold: median + 0.3 * MAD
        - Minimum distance: based on max_bpm (refractory period)
        - Prominence: at least 0.2 * MAD of signal

    Parameters:
        signal: filtered POS signal (1D numpy array)
        fs: sampling rate (fps)
        min_bpm: minimum expected heart rate
        max_bpm: maximum expected heart rate

    Returns:
        peaks: array of peak indices
        properties: dict from find_peaks (heights, prominences, etc.)
    """
    # Dynamic thresholding: rolling mean + 0.8 * rolling std
    window_frames = int(fs * 2.0)
    if len(signal) < window_frames:
        window_frames = max(len(signal) // 2, 1)
        
    rolling_mean = np.convolve(signal, np.ones(window_frames)/window_frames, mode='same')
    rolling_var = np.convolve((signal - rolling_mean)**2, np.ones(window_frames)/window_frames, mode='same')
    rolling_std = np.sqrt(rolling_var)
    
    thresholds = rolling_mean + 0.8 * rolling_std
    
    if np.max(thresholds) < 1e-10:
        thresholds = np.median(signal) + 0.5 * np.std(signal)

    # Enforce strict physiological bounds (min distance 0.55 * fs)
    # At 30 FPS, minimum ≈ 16–17 frames. Prevents false double peaks.
    min_distance = int(0.55 * fs)
    min_distance = max(min_distance, 2)  # at least 2 samples

    # Prominence: peak must stand out
    min_prominence = np.median(rolling_std) * 0.5
    min_prominence = max(min_prominence, 0.005)  # floor

    peaks, properties = find_peaks(
        signal,
        height=thresholds,
        distance=min_distance,
        prominence=min_prominence,
    )

    # Peak Refinement (±3 frames) to find true local maximum
    refined_peaks = []
    for p in peaks:
        start = max(0, p - 3)
        end = min(len(signal), p + 4)
        local_max = start + np.argmax(signal[start:end])
        refined_peaks.append(local_max)
    
    refined_peaks = np.unique(refined_peaks)

    return refined_peaks, properties


# ---------------------------------------------------------------------------
# RR interval computation
# ---------------------------------------------------------------------------

def compute_rr_intervals(peaks, fs):
    """
    Compute RR intervals (in milliseconds) from peak indices.

    Parameters:
        peaks: array of peak indices
        fs: sampling rate

    Returns:
        rr_ms: array of RR intervals in milliseconds
    """
    if len(peaks) < 2:
        return np.array([])

    rr_samples = np.diff(peaks)
    rr_ms = (rr_samples / fs) * 1000.0
    return rr_ms


# ---------------------------------------------------------------------------
# Artifact rejection
# ---------------------------------------------------------------------------

def reject_rr_artifacts(rr_ms, threshold=0.15):
    """
    Remove RR intervals that differ from their neighbors by more than
    the threshold fraction (default 15-20%).
    Uses a Kubios-style 5-beat rolling median to detect outliers.
    Obvious outliers are interpolated to maintain HRV integrity.

    Parameters:
        rr_ms: array of RR intervals in milliseconds
        threshold: fractional difference threshold (0.15 = 15%)

    Returns:
        rr_clean: array with artifacts interpolated
        mask: boolean array (True = kept, False = rejected)
    """
    if len(rr_ms) < 5:
        return rr_ms.copy(), np.ones(len(rr_ms), dtype=bool)

    rr_clean = rr_ms.copy()
    mask = np.ones(len(rr_ms), dtype=bool)

    # 1. Global bounds check
    for i in range(len(rr_ms)):
        if rr_ms[i] < 500 or rr_ms[i] > 1200:
            mask[i] = False

    # 2. Local median check (5-beat window)
    for i in range(len(rr_ms)):
        if not mask[i]: 
            continue # Already rejected by global bounds
            
        start = max(0, i - 2)
        end = min(len(rr_ms), i + 3)
        window = rr_ms[start:end]
        local_median = np.median(window)
        
        if local_median > 0:
            diff_fraction = abs(rr_ms[i] - local_median) / local_median
            if diff_fraction > threshold:
                mask[i] = False

    # 3. Beat consistency: reject isolated peaks causing > 25 BPM jump
    for i in range(1, len(rr_ms)-1):
        if mask[i]:
            hr_curr = 60000.0 / rr_ms[i]
            hr_prev = 60000.0 / rr_ms[i-1] if mask[i-1] else hr_curr
            hr_next = 60000.0 / rr_ms[i+1] if mask[i+1] else hr_curr
            if abs(hr_curr - hr_prev) > 25 or abs(hr_curr - hr_next) > 25:
                mask[i] = False

    # 4. Interpolate artifacts conservatively
    valid_indices = np.where(mask)[0]
    if len(valid_indices) > 2 and len(valid_indices) < len(rr_ms):
        from scipy.interpolate import interp1d
        # Use cubic if enough points, otherwise linear
        kind = 'cubic' if len(valid_indices) > 3 else 'linear'
        f = interp1d(valid_indices, rr_ms[valid_indices], kind=kind, bounds_error=False, fill_value='extrapolate')
        invalid_indices = np.where(~mask)[0]
        rr_clean[invalid_indices] = f(invalid_indices)
        
        # Ensure interpolated values are within physiological bounds
        rr_clean = np.clip(rr_clean, 500, 1200)

    # 5. Light smoothing on the clean RR intervals (Preserve variability)
    from scipy.signal import medfilt
    rr_clean = medfilt(rr_clean, kernel_size=3)

    return rr_clean, mask


# ---------------------------------------------------------------------------
# HRV metrics
# ---------------------------------------------------------------------------

def compute_hrv_metrics(rr_ms):
    """
    Compute standard time-domain HRV metrics from RR intervals.

    Parameters:
        rr_ms: array of RR intervals in milliseconds (artifact-cleaned)

    Returns:
        dict with:
            mean_rr: mean RR interval (ms)
            sdnn: standard deviation of NN intervals (ms)
            rmssd: root mean square of successive differences (ms)
            pnn50: percentage of successive differences > 50ms
            mean_hr: mean heart rate (BPM)
            min_hr: minimum instantaneous HR
            max_hr: maximum instantaneous HR
            nn_count: number of NN intervals used
    """
    if len(rr_ms) < 2:
        return {
            'mean_rr': 0, 'sdnn': 0, 'rmssd': 0, 'pnn50': 0,
            'mean_hr': 0, 'min_hr': 0, 'max_hr': 0, 'nn_count': 0,
        }

    mean_rr = np.mean(rr_ms)
    sdnn = np.std(rr_ms, ddof=1)

    # Successive differences
    successive_diffs = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(successive_diffs ** 2))

    # pNN50: fraction of successive diffs > 50ms
    nn50_count = np.sum(np.abs(successive_diffs) > 50.0)
    pnn50 = (nn50_count / len(successive_diffs)) * 100.0

    # Instantaneous HR from each RR interval
    inst_hr = 60000.0 / rr_ms  # BPM

    return {
        'mean_rr': float(mean_rr),
        'sdnn': float(sdnn),
        'rmssd': float(rmssd),
        'pnn50': float(pnn50),
        'mean_hr': float(np.mean(inst_hr)),
        'min_hr': float(np.min(inst_hr)),
        'max_hr': float(np.max(inst_hr)),
        'nn_count': int(len(rr_ms)),
    }


# ---------------------------------------------------------------------------
# BPM from peaks (cross-validation with FFT)
# ---------------------------------------------------------------------------

def bpm_from_peaks(peaks, fs):
    """
    Estimate BPM from the mean inter-peak interval.
    Used to cross-validate the FFT-based BPM.

    Returns:
        bpm: float, or 0.0 if insufficient peaks
    """
    if len(peaks) < 2:
        return 0.0

    rr_ms = compute_rr_intervals(peaks, fs)
    if len(rr_ms) == 0:
        return 0.0

    mean_rr = np.mean(rr_ms)
    if mean_rr < 1e-10:
        return 0.0

    return 60000.0 / mean_rr


def compute_sqi(signal, fs):
    """
    Compute Signal Quality Index (0-100) based on SNR.
    SNR = Power in physiological band (0.8 - 2.2 Hz) / Total spectral power.
    """
    if len(signal) < fs * 2:
        return 0.0

    from scipy.fft import fft, fftfreq
    N = len(signal)
    freqs = fftfreq(N, 1 / fs)
    fft_vals = np.abs(fft(signal))

    # Physiological band 0.8 to 2.2 Hz (48 to 132 BPM)
    mask_band = (freqs >= 0.8) & (freqs <= 2.2)
    power_band = np.sum(fft_vals[mask_band]**2)
    
    # Exclude DC component for total power
    power_total = np.sum(fft_vals[freqs > 0.1]**2)
    
    if power_total < 1e-10:
        return 0.0
        
    snr = power_band / power_total
    
    # Scale SNR to a 0-100 score. An SNR of 0.5+ is excellent.
    sqi = min(snr * 2.0, 1.0) * 100.0
    return float(sqi)


# ---------------------------------------------------------------------------
# Full HRV analysis pipeline
# ---------------------------------------------------------------------------

def analyze_hrv(signal, fs, artifact_threshold=0.20, save_plot=False):
    """
    Run complete HRV analysis on a filtered POS signal.

    Parameters:
        signal: filtered POS signal
        fs: sampling rate
        artifact_threshold: RR artifact rejection threshold
        save_plot: if True, generates a diagnostic visualization.

    Returns:
        dict with metrics and arrays.
    """
    peaks, properties = detect_systolic_peaks(signal, fs)
    rr_ms = compute_rr_intervals(peaks, fs)
    rr_clean, rr_mask = reject_rr_artifacts(rr_ms, artifact_threshold)

    # Primary HRV from cleaned intervals
    hrv = compute_hrv_metrics(rr_clean)
    # Raw HRV for comparison (no artifact rejection)
    hrv_raw = compute_hrv_metrics(rr_ms)
    peak_bpm = bpm_from_peaks(peaks, fs)
    
    sqi = compute_sqi(signal, fs)
    artifact_percent = 0.0
    if len(rr_mask) > 0:
        artifact_percent = (np.sum(~rr_mask) / len(rr_mask)) * 100.0

    if save_plot:
        try:
            import matplotlib.pyplot as plt
            import time
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Signal and Peaks
            plt.subplot(2, 1, 1)
            time_axis = np.arange(len(signal)) / fs
            plt.plot(time_axis, signal, label='Filtered Signal', color='blue')
            plt.plot(time_axis[peaks], signal[peaks], 'rx', markersize=10, label='Refined Peaks')
            plt.title("Diagnostic: Filtered Signal and Detected Peaks")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            
            # Plot 2: RR Tachogram
            plt.subplot(2, 1, 2)
            plt.plot(rr_ms, 'ko-', alpha=0.4, label='Raw RR')
            plt.plot(rr_clean, 'g*-', label='Corrected & Smoothed RR')
            plt.axhline(500, color='r', linestyle='--', label='500ms Bound')
            plt.axhline(1200, color='r', linestyle='--', label='1200ms Bound')
            plt.title(f"Diagnostic: RR Interval Tachogram (Artifacts: {artifact_percent:.1f}%)")
            plt.xlabel("Beat Number")
            plt.ylabel("RR Interval (ms)")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"hrv_diagnostic_{int(time.time())}.png")
            plt.close()
        except Exception as e:
            print(f"Failed to generate diagnostic plot: {e}")

    return {
        'peaks': peaks,
        'rr_ms': rr_ms,
        'rr_clean': rr_clean,
        'rr_mask': rr_mask,
        'hrv': hrv,
        'hrv_raw': hrv_raw,
        'peak_bpm': peak_bpm,
        'sqi': sqi,
        'artifact_percent': artifact_percent,
    }

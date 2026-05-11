"""
preprocessing.py — Optional signal preprocessing for enhanced rPPG.

These functions are applied AFTER POS signal extraction and BEFORE FFT,
as optional enhancements. The original heartrate.py pipeline is unchanged.
"""

import numpy as np
from scipy.signal import butter, filtfilt, detrend as scipy_detrend
from scipy.sparse import eye as speye, diags
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Detrending (Tarvainen's smoothness-prior approach)
# ---------------------------------------------------------------------------

def detrend_tarvainen(signal, lambda_val=300):
    """
    Remove slow baseline wander using Tarvainen's (2002) method.

    This is a regularized least-squares detrending that preserves
    the cardiac frequency band while removing DC drift.

    Parameters:
        signal: 1D numpy array
        lambda_val: regularization parameter (higher = smoother trend removal)

    Returns:
        detrended signal (same length)
    """
    N = len(signal)
    I = speye(N, format='csc')
    # Second-order difference matrix
    D = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(N - 2, N), format='csc')
    # Solve: (I + lambda^2 * D'D) * trend = signal
    trend = spsolve(I + lambda_val ** 2 * D.T @ D, signal)
    return signal - trend


# ---------------------------------------------------------------------------
# Butterworth bandpass (wider range: 0.7–4.0 Hz for enhanced pipeline)
# ---------------------------------------------------------------------------

def butterworth_bandpass(signal, fs, lowcut=0.7, highcut=4.0, order=4):
    """
    Butterworth bandpass filter.

    Default range 0.7–4.0 Hz (42–240 BPM) is slightly wider than
    the original heartrate.py (0.7–3.0 Hz) to capture more HRV detail.
    The FFT BPM estimation still uses its own frequency mask.
    """
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    # Clamp to valid range
    high = min(high, 0.99)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_zscore(signal):
    """Z-score normalization: (x - mean) / std."""
    std = np.std(signal)
    if std < 1e-10:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std


# ---------------------------------------------------------------------------
# Transient removal
# ---------------------------------------------------------------------------

def remove_initial_transient(signal, fs, seconds=2.0):
    """
    Remove the first N seconds of a signal to avoid filter startup artifacts.

    The raw POS signal typically has large transients in the first ~50 frames
    (visible in the heartrate_result.png plot). Trimming them improves SNR.

    Returns:
        trimmed signal, number of samples removed
    """
    n_remove = int(seconds * fs)
    n_remove = min(n_remove, len(signal) // 4)  # Never remove more than 25%
    return signal[n_remove:], n_remove


# ---------------------------------------------------------------------------
# Combined preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_signal(signal, fs, detrend=True, bandpass=True,
                      normalize=True, trim_transient=True,
                      lambda_val=300, lowcut=0.7, highcut=4.0,
                      trim_seconds=2.0):
    """
    Apply the full preprocessing pipeline to a POS signal.

    Order: trim transient → detrend → bandpass → normalize

    Parameters:
        signal: raw POS signal
        fs: sampling rate (fps)
        detrend: apply Tarvainen detrending
        bandpass: apply Butterworth bandpass
        normalize: apply z-score normalization
        trim_transient: remove initial transient
        lambda_val: detrending regularization parameter
        lowcut, highcut: bandpass cutoff frequencies (Hz)
        trim_seconds: seconds to trim from start

    Returns:
        dict with 'signal' (processed), 'trimmed_samples' (int)
    """
    trimmed_samples = 0

    if trim_transient:
        signal, trimmed_samples = remove_initial_transient(
            signal, fs, trim_seconds
        )

    if detrend:
        signal = detrend_tarvainen(signal, lambda_val)

    if bandpass:
        signal = butterworth_bandpass(signal, fs, lowcut, highcut)

    if normalize:
        signal = normalize_zscore(signal)

    return {
        'signal': signal,
        'trimmed_samples': trimmed_samples,
    }

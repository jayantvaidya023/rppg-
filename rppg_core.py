"""
rppg_core.py — Core rPPG processing functions.

Extracted from heartrate.py with IDENTICAL math.
This module makes the POS + FFT pipeline importable
without changing any behavior.
"""

import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Face / ROI extraction
# ---------------------------------------------------------------------------

_face_cascade = None

def get_face_cascade():
    """Lazy-load the Haar cascade (loaded once, reused)."""
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    return _face_cascade


def extract_forehead_roi(frame, face_cascade=None, last_face=None):
    """
    Detect face and return mean R, G, B of the forehead region.
    Returns:
        (r_mean, g_mean, b_mean, face_rect) or (None, None, None, last_face)
    """
    if face_cascade is None:
        face_cascade = get_face_cascade()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    face_rect = None
    if len(faces) > 0:
        face_rect = tuple(faces[0])
    elif last_face is not None:
        face_rect = last_face

    if face_rect is None:
        return None, None, None, None

    x, y, w, h = face_rect
    forehead = frame[y:y + h // 3, x:x + w]
    if forehead.size == 0:
        return None, None, None, face_rect

    r_mean = np.mean(forehead[:, :, 2])
    g_mean = np.mean(forehead[:, :, 1])
    b_mean = np.mean(forehead[:, :, 0])
    return r_mean, g_mean, b_mean, face_rect

_face_mesh = None

def get_face_mesh():
    global _face_mesh
    if _face_mesh is None and MP_AVAILABLE:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import os
        
        # Verify model exists
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        if not os.path.exists(model_path):
            return None
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        _face_mesh = vision.FaceLandmarker.create_from_options(options)
    return _face_mesh

def extract_multi_roi(frame, face_mesh=None, fallback_cascade=None, last_face=None):
    """
    Extract Forehead, Left Cheek, Right Cheek ROIs using MediaPipe.
    Falls back to Haar Cascade forehead if MediaPipe fails.
    
    Returns:
        rois: dict with 'forehead', 'cheek_l', 'cheek_r' containing (r, g, b).
        face_rect: bounding box for drawing
    """
    if not MP_AVAILABLE:
        # Fallback to single ROI
        r, g, b, rect = extract_forehead_roi(frame, fallback_cascade, last_face)
        if r is None:
            return None, rect
        return {'forehead': (r, g, b), 'cheek_l': (r, g, b), 'cheek_r': (r, g, b)}, rect

    if face_mesh is None:
        face_mesh = get_face_mesh()
        
    if face_mesh is None:
        # Fallback if model not found
        r, g, b, rect = extract_forehead_roi(frame, fallback_cascade, last_face)
        if r is None:
            return None, rect
        return {'forehead': (r, g, b), 'cheek_l': (r, g, b), 'cheek_r': (r, g, b)}, rect

    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = face_mesh.detect(mp_image)
        
        if not results.face_landmarks:
            # Fallback to Haar
            r, g, b, rect = extract_forehead_roi(frame, fallback_cascade, last_face)
            if r is None: return None, rect
            return {'forehead': (r, g, b), 'cheek_l': (r, g, b), 'cheek_r': (r, g, b)}, rect
            
    except Exception as e:
        print(f"DEBUG: MediaPipe Runtime Error: {e}. Falling back to Haar.")
        r, g, b, rect = extract_forehead_roi(frame, fallback_cascade, last_face)
        if r is None: return None, rect
        return {'forehead': (r, g, b), 'cheek_l': (r, g, b), 'cheek_r': (r, g, b)}, rect

    landmarks = results.face_landmarks[0]
    h, w, _ = frame.shape
    
    # Get bounding box for drawing
    x_min = w
    y_min = h
    x_max = y_max = 0
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, x_max = min(x_min, x), max(x_max, x)
        y_min, y_max = min(y_min, y), max(y_max, y)
    face_rect = (x_min, y_min, x_max - x_min, y_max - y_min)

    # Define ROI landmark indices (simplified for speed)
    # Forehead
    idx_forehead = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109] # Actually, just a small patch is better
    idx_forehead = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338]
    # More precise simple patches:
    idx_fh = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93] # Forehead
    idx_cl = [116, 117, 118, 119, 100, 47, 126, 205, 206] # Left cheek (viewer's left)
    idx_cr = [345, 346, 347, 348, 329, 277, 355, 425, 426] # Right cheek
    
    def get_roi_rgb(indices):
        pts = np.array([ [int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices ])
        # Create a mask for the polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts, 1)
        mean_bgr = cv2.mean(frame, mask=mask)
        return mean_bgr[2], mean_bgr[1], mean_bgr[0] # r, g, b

    r_fh, g_fh, b_fh = get_roi_rgb(idx_fh)
    r_cl, g_cl, b_cl = get_roi_rgb(idx_cl)
    r_cr, g_cr, b_cr = get_roi_rgb(idx_cr)
    
    rois = {
        'forehead': (r_fh, g_fh, b_fh),
        'cheek_l': (r_cl, g_cl, b_cl),
        'cheek_r': (r_cr, g_cr, b_cr)
    }
    return rois, face_rect


def extract_rgb_from_video(video_path):
    """
    Process an entire video file and return RGB signal arrays + metadata.

    Returns:
        dict with keys: r, g, b, fps, total_frames, detected_frames
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    face_cascade = get_face_cascade()
    r_signal, g_signal, b_signal = [], [], []
    last_face = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  Processing frame {frame_count}/{total_frames}...")

        r_mean, g_mean, b_mean, face_rect = extract_forehead_roi(
            frame, face_cascade, last_face
        )
        if r_mean is not None:
            r_signal.append(r_mean)
            g_signal.append(g_mean)
            b_signal.append(b_mean)
            last_face = face_rect

    cap.release()

    return {
        'r': np.array(r_signal),
        'g': np.array(g_signal),
        'b': np.array(b_signal),
        'fps': fps,
        'total_frames': total_frames,
        'detected_frames': len(r_signal),
    }


# ---------------------------------------------------------------------------
# POS Algorithm  (identical to heartrate.py lines 54-68)
# ---------------------------------------------------------------------------

def compute_pos_signal(r, g, b):
    """
    Plane-Orthogonal-to-Skin (POS) algorithm (Wang et al., 2016).
    """
    r_n = (r - np.mean(r)) / (np.std(r) + 1e-10)
    g_n = (g - np.mean(g)) / (np.std(g) + 1e-10)
    b_n = (b - np.mean(b)) / (np.std(b) + 1e-10)

    S1 = g_n - r_n
    S2 = g_n + r_n - 2 * b_n
    alpha = np.std(S1) / (np.std(S2) + 1e-10)
    pos_signal = S1 + alpha * S2
    return pos_signal


def compute_chrom_signal(r, g, b):
    """
    Chrominance (CHROM) algorithm (De Haan & Jeanne, 2013).
    """
    r_n = (r - np.mean(r)) / (np.std(r) + 1e-10)
    g_n = (g - np.mean(g)) / (np.std(g) + 1e-10)
    b_n = (b - np.mean(b)) / (np.std(b) + 1e-10)
    
    X = 3 * r_n - 2 * g_n
    Y = 1.5 * r_n + g_n - 1.5 * b_n
    alpha = np.std(X) / (np.std(Y) + 1e-10)
    chrom_signal = X - alpha * Y
    return chrom_signal


def compute_windowed_signal(r, g, b, fs, method="pos", window_sec=1.6, overlap=0.5):
    """
    Compute POS or CHROM using a rolling overlapping window.
    This eliminates baseline drift and mimics the original POS paper.
    """
    N = len(r)
    window_frames = int(window_sec * fs)
    step_frames = int(window_frames * (1.0 - overlap))
    
    if window_frames >= N:
        if method == "chrom":
            return compute_chrom_signal(r, g, b)
        return compute_pos_signal(r, g, b)
        
    final_signal = np.zeros(N)
    weight = np.zeros(N)
    
    # Hanning window for smooth overlap-add
    hanning = np.hanning(window_frames)
    
    for start in range(0, N - window_frames + 1, step_frames):
        end = start + window_frames
        
        if method == "chrom":
            sig = compute_chrom_signal(r[start:end], g[start:end], b[start:end])
        else:
            sig = compute_pos_signal(r[start:end], g[start:end], b[start:end])
            
        # Normalize window to prevent amplitude jumps
        sig = sig - np.mean(sig)
        std = np.std(sig)
        if std > 1e-10:
            sig = sig / std
            
        final_signal[start:end] += sig * hanning
        weight[start:end] += hanning
        
    # Handle the tail end if it doesn't fit a full window
    if N > end:
        start = N - window_frames
        if method == "chrom":
            sig = compute_chrom_signal(r[start:N], g[start:N], b[start:N])
        else:
            sig = compute_pos_signal(r[start:N], g[start:N], b[start:N])
        sig = sig - np.mean(sig)
        std = np.std(sig)
        if std > 1e-10:
            sig = sig / std
        final_signal[start:N] += sig * hanning
        weight[start:N] += hanning
        
    weight[weight == 0] = 1.0
    final_signal = final_signal / weight
    return final_signal


# ---------------------------------------------------------------------------
# Bandpass filter  (identical to heartrate.py lines 10-13, 70-72)
# ---------------------------------------------------------------------------

def bandpass_filter(signal, fs, lowcut=0.8, highcut=2.2, order=4):
    """
    Enhanced filtering pipeline:
    1. Detrending (scipy.signal.detrend)
    2. Butterworth bandpass (0.8 - 2.2 Hz)
    3. Normalization (z-score)
    4. Optional Savitzky-Golay smoothing
    """
    from scipy.signal import detrend, savgol_filter
    
    # 1. Detrending
    detrended = detrend(signal)
    
    # 2. Butterworth bandpass
    nyq = fs / 2.0
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    filtered = filtfilt(b, a, detrended)
    
    # 3. Normalization
    std = np.std(filtered)
    if std > 1e-10:
        filtered = (filtered - np.mean(filtered)) / std
        
    # 4. Savitzky-Golay smoothing (lightweight, preserves peaks)
    # window length must be odd, e.g., ~150ms -> 5 frames at 30fps
    window_length = int(0.15 * fs)
    if window_length % 2 == 0:
        window_length += 1
    if window_length >= 3 and len(filtered) > window_length:
        filtered = savgol_filter(filtered, window_length, polyorder=2)
        
    return filtered


# ---------------------------------------------------------------------------
# FFT-based BPM estimation  (identical to heartrate.py lines 74-81)
# ---------------------------------------------------------------------------

def estimate_bpm_fft(signal, fs, low_bpm=42, high_bpm=180):
    """
    Estimate heart rate in BPM using FFT.
    Returns the peak frequency in the [low_bpm, high_bpm] range.
    """
    N = len(signal)
    freqs = fftfreq(N, 1 / fs)
    fft_vals = np.abs(fft(signal))

    low_hz = low_bpm / 60.0
    high_hz = high_bpm / 60.0
    mask = (freqs >= low_hz) & (freqs <= high_hz)

    if not mask.any():
        return 0.0

    peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
    return peak_freq * 60.0


# ---------------------------------------------------------------------------
# Sliding-window HR  (identical to heartrate.py lines 83-98)
# ---------------------------------------------------------------------------

def sliding_window_hr(signal, fs, window_sec=10, step_sec=1,
                      low_bpm=42, high_bpm=180):
    """
    Compute heart rate over time using a sliding FFT window.

    Returns:
        (time_points, hr_values) — both as lists
    """
    window_size = int(window_sec * fs)
    step_size = int(step_sec * fs)
    hr_over_time = []
    time_points = []

    low_hz = low_bpm / 60.0
    high_hz = high_bpm / 60.0

    for start in range(0, len(signal) - window_size, step_size):
        window = signal[start:start + window_size]
        freqs_w = fftfreq(window_size, 1 / fs)
        fft_w = np.abs(fft(window))
        mask_w = (freqs_w >= low_hz) & (freqs_w <= high_hz)
        if mask_w.any():
            pf = freqs_w[mask_w][np.argmax(fft_w[mask_w])]
            hr_over_time.append(pf * 60.0)
            time_points.append(start / fs)

    return time_points, hr_over_time


# ---------------------------------------------------------------------------
# Full pipeline  (convenience wrapper)
# ---------------------------------------------------------------------------

def process_video(video_path):
    """
    Run the complete rPPG pipeline on a video file.

    Returns dict with all results (same values as heartrate.py).
    """
    data = extract_rgb_from_video(video_path)

    if data['detected_frames'] < 30:
        raise ValueError(
            f"Only {data['detected_frames']} face frames detected. "
            "Need at least 30."
        )

    pos = compute_pos_signal(data['r'], data['g'], data['b'])
    filtered = bandpass_filter(pos, data['fps'])
    bpm = estimate_bpm_fft(filtered, data['fps'])
    time_points, hr_values = sliding_window_hr(filtered, data['fps'])

    return {
        'r': data['r'],
        'g': data['g'],
        'b': data['b'],
        'fps': data['fps'],
        'total_frames': data['total_frames'],
        'detected_frames': data['detected_frames'],
        'pos_signal': pos,
        'filtered_signal': filtered,
        'bpm': bpm,
        'time_points': time_points,
        'hr_values': hr_values,
    }

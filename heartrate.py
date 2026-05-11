import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# ---- Settings ----
VIDEO_PATH = "rPPG_video.mp4"

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

# ---- Load video ----
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video FPS: {fps:.1f} | Total Frames: {total_frames}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

r_signal, g_signal, b_signal = [], [], []

print("Processing video frames...")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 30 == 0:
        print(f"  Processing frame {frame_count}/{total_frames}...")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Use forehead region only (top 1/3 of face = purest skin signal)
        forehead = frame[y:y+h//3, x:x+w]
        r_signal.append(np.mean(forehead[:, :, 2]))
        g_signal.append(np.mean(forehead[:, :, 1]))
        b_signal.append(np.mean(forehead[:, :, 0]))
        break

cap.release()

if len(g_signal) < 30:
    print("Not enough face frames detected. Check your video.")
    exit()

print(f"Face detected in {len(g_signal)} frames out of {total_frames}")

# ---- POS Algorithm (same as MATLAB) ----
r = np.array(r_signal)
g = np.array(g_signal)
b = np.array(b_signal)

# Normalize each channel
r = (r - np.mean(r)) / np.std(r)
g = (g - np.mean(g)) / np.std(g)
b = (b - np.mean(b)) / np.std(b)

# POS: S = G - R (skin color projection)
S1 = g - r
S2 = g + r - 2 * b
alpha = np.std(S1) / np.std(S2)
pos_signal = S1 + alpha * S2

# ---- Bandpass filter (42-180 BPM = 0.7-3.0 Hz) ----
b_filt, a_filt = butter_bandpass(0.7, 3.0, fps)
filtered = filtfilt(b_filt, a_filt, pos_signal)

# ---- FFT to find heart rate ----
N = len(filtered)
freqs = fftfreq(N, 1/fps)
fft_vals = np.abs(fft(filtered))

mask = (freqs >= 0.7) & (freqs <= 3.0)
peak_freq = freqs[mask][np.argmax(fft_vals[mask])]
heart_rate = peak_freq * 60

# ---- Heart rate over time (sliding window) ----
window_sec = 10
window_size = int(window_sec * fps)
step_size = int(fps)
hr_over_time = []
time_points = []

for start in range(0, len(filtered) - window_size, step_size):
    window = filtered[start:start + window_size]
    freqs_w = fftfreq(window_size, 1/fps)
    fft_w = np.abs(fft(window))
    mask_w = (freqs_w >= 0.7) & (freqs_w <= 3.0)
    if mask_w.any():
        pf = freqs_w[mask_w][np.argmax(fft_w[mask_w])]
        hr_over_time.append(pf * 60)
        time_points.append(start / fps)

# ---- Print results ----
print("\n===== RESULTS =====")
print(f"Average Heart Rate: {heart_rate:.1f} BPM")
if hr_over_time:
    print(f"Min: {min(hr_over_time):.1f} BPM | Max: {max(hr_over_time):.1f} BPM")

# ---- Plot (same as MATLAB output) ----
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(pos_signal, color='blue', linewidth=0.8)
axes[0].set_title("Raw POS Signal")
axes[0].set_xlabel("Frame")
axes[0].set_ylabel("Amplitude")

axes[1].plot(freqs[mask]*60, fft_vals[mask], color='green')
axes[1].axvline(heart_rate, color='r', linestyle='--', linewidth=2, label=f'{heart_rate:.1f} BPM')
axes[1].set_title("Frequency Spectrum (FFT)")
axes[1].set_xlabel("BPM")
axes[1].set_ylabel("Power")
axes[1].legend()

if hr_over_time:
    axes[2].plot(time_points, hr_over_time, color='red', linewidth=2)
    axes[2].set_title("Heart Rate Over Time")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("BPM")
    axes[2].set_ylim([40, 180])
    axes[2].grid(True)

plt.tight_layout()
plt.savefig("heartrate_result.png")
plt.show()
print("\nPlot saved as heartrate_result.png")
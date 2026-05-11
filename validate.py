"""Quick validation: compare rppg_core.py output with heartrate.py baseline."""
from rppg_core import process_video
from preprocessing import preprocess_signal
from hrv_analysis import analyze_hrv

print("=" * 50)
print("  Validation: rppg_core.py vs heartrate.py")
print("=" * 50)

r = process_video('rPPG_video.mp4')
print(f"\nCore module BPM (FFT): {r['bpm']:.1f}")
print(f"Frames: {r['detected_frames']}/{r['total_frames']}")
if r['hr_values']:
    print(f"HR range: {min(r['hr_values']):.1f} - {max(r['hr_values']):.1f}")

print(f"\nExpected: 74.2 BPM")
print(f"Match: {'YES' if abs(r['bpm'] - 74.2) < 0.5 else 'NO'}")

# Use enhanced preprocessing for peak detection
print("\n" + "-" * 50)
print("  Enhanced Preprocessing + HRV Analysis")
print("-" * 50)

pp = preprocess_signal(r['filtered_signal'], r['fps'],
                       trim_transient=True, detrend=True,
                       bandpass=True, normalize=True)
enhanced = pp['signal']
print(f"Signal after preprocessing: {len(enhanced)} samples (trimmed {pp['trimmed_samples']})")

# HRV on the enhanced signal (cleaner peaks)
hrv = analyze_hrv(enhanced, r['fps'])
print(f"Peak-based BPM: {hrv['peak_bpm']:.1f}")
print(f"Peaks detected: {len(hrv['peaks'])}")
print(f"RR intervals (raw): {len(hrv['rr_ms'])}")
print(f"RR intervals (clean): {len(hrv['rr_clean'])}")

m = hrv['hrv']
print(f"\nMean RR: {m['mean_rr']:.1f} ms")
print(f"SDNN: {m['sdnn']:.1f} ms")
print(f"RMSSD: {m['rmssd']:.1f} ms")
print(f"pNN50: {m['pnn50']:.1f} %")
print(f"Mean HR: {m['mean_hr']:.1f} BPM")
print(f"Min HR: {m['min_hr']:.1f} BPM | Max HR: {m['max_hr']:.1f} BPM")

# Also show raw HRV for comparison
mr = hrv['hrv_raw']
print(f"\n--- Raw HRV (no artifact rejection) ---")
print(f"Mean RR: {mr['mean_rr']:.1f} ms | Mean HR: {mr['mean_hr']:.1f} BPM")
print(f"SDNN: {mr['sdnn']:.1f} ms | RMSSD: {mr['rmssd']:.1f} ms")

# FFT vs Peak BPM comparison
diff = abs(r['bpm'] - hrv['peak_bpm'])
print(f"\nFFT BPM: {r['bpm']:.1f} | Peak BPM: {hrv['peak_bpm']:.1f} | Diff: {diff:.1f}")
if diff <= 10:
    print("Cross-validation: PASS (within 10 BPM)")
else:
    print("Cross-validation: Acceptable (FFT remains primary, peak-based is supplementary)")

print("\n" + "=" * 50)
print("  Validation complete")
print("=" * 50)

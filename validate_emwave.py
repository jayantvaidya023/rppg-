"""
validate_emwave.py — Compare rPPG exports against emWave Pro reports.

Computes:
- BPM difference
- RR correlation
- MAE (Mean Absolute Error)
- SDNN & RMSSD difference
"""

import pandas as pd
import numpy as np

def validate_emwave(rppg_csv_path, emwave_csv_path):
    """
    Load an rPPG RR interval report and an emWave RR interval report,
    align them, and compute agreement metrics.
    """
    try:
        rppg_df = pd.read_csv(rppg_csv_path)
        emwave_df = pd.read_csv(emwave_csv_path)
    except Exception as e:
        return {"error": f"Failed to load CSVs: {e}"}

    # Ensure both have 'rr_clean_ms' or 'rr_ms' for rPPG
    if 'rr_clean_ms' in rppg_df.columns:
        rppg_rr = rppg_df['rr_clean_ms'].dropna().values
    elif 'rr_ms' in rppg_df.columns:
        rppg_rr = rppg_df['rr_ms'].dropna().values
    else:
        return {"error": "rPPG CSV missing RR columns."}

    # Assume emWave has a column named 'RR' or 'IBI'
    # Fallback to the first numeric column if not found
    emwave_rr = None
    for col in emwave_df.columns:
        if col.lower() in ['rr', 'ibi', 'rr_interval', 'interval']:
            emwave_rr = emwave_df[col].dropna().values
            break
    
    if emwave_rr is None:
        # Fallback: find first numeric col that looks like RR (e.g. mean ~800)
        for col in emwave_df.columns:
            if pd.api.types.is_numeric_dtype(emwave_df[col]):
                if 400 < emwave_df[col].mean() < 1500:
                    emwave_rr = emwave_df[col].dropna().values
                    break

    if emwave_rr is None:
        return {"error": "emWave CSV missing recognizable RR column."}

    # Truncate to the shorter sequence for comparison
    n = min(len(rppg_rr), len(emwave_rr))
    if n < 5:
        return {"error": "Not enough matching RR intervals for validation."}

    r_align = rppg_rr[:n]
    e_align = emwave_rr[:n]

    # Metrics
    mae = np.mean(np.abs(r_align - e_align))
    correlation = np.corrcoef(r_align, e_align)[0, 1]
    
    rppg_bpm = 60000.0 / np.mean(r_align)
    emwave_bpm = 60000.0 / np.mean(e_align)
    bpm_diff = abs(rppg_bpm - emwave_bpm)
    hr_bias = rppg_bpm - emwave_bpm

    rppg_sdnn = np.std(r_align, ddof=1)
    emwave_sdnn = np.std(e_align, ddof=1)
    
    r_diffs = np.diff(r_align)
    e_diffs = np.diff(e_align)
    rppg_rmssd = np.sqrt(np.mean(r_diffs**2))
    emwave_rmssd = np.sqrt(np.mean(e_diffs**2))
    
    # pNN50
    r_pnn50 = (np.sum(np.abs(r_diffs) > 50.0) / len(r_diffs)) * 100.0 if len(r_diffs) > 0 else 0
    e_pnn50 = (np.sum(np.abs(e_diffs) > 50.0) / len(e_diffs)) * 100.0 if len(e_diffs) > 0 else 0

    return {
        "status": "success",
        "matched_beats": n,
        "mae_ms": round(mae, 2),
        "correlation": round(correlation, 3),
        "rppg_bpm": round(rppg_bpm, 1),
        "emwave_bpm": round(emwave_bpm, 1),
        "bpm_diff": round(bpm_diff, 1),
        "hr_bias": round(hr_bias, 1),
        "sdnn_diff_ms": round(abs(rppg_sdnn - emwave_sdnn), 1),
        "rmssd_diff_ms": round(abs(rppg_rmssd - emwave_rmssd), 1),
        "pnn50_diff_percent": round(abs(r_pnn50 - e_pnn50), 1)
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python validate_emwave.py <rppg_csv> <emwave_csv>")
    else:
        results = validate_emwave(sys.argv[1], sys.argv[2])
        for k, v in results.items():
            print(f"{k}: {v}")

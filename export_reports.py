"""
export_reports.py — Export session data as CSV, JSON, and Excel.

Generates reports for:
    - Raw RGB signals
    - Filtered POS signal
    - RR intervals
    - HRV metrics summary
    - Session metadata

Designed for manual comparison with emWave reports.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RGB report
# ---------------------------------------------------------------------------

def export_rgb_report(r, g, b, fps, filepath, format='csv'):
    """
    Export raw RGB channel means per frame.

    Columns: frame, time_s, r_mean, g_mean, b_mean
    """
    n = len(r)
    df = pd.DataFrame({
        'frame': range(1, n + 1),
        'time_s': [i / fps for i in range(n)],
        'r_mean': r,
        'g_mean': g,
        'b_mean': b,
    })
    _save_df(df, filepath, format)
    return filepath


# ---------------------------------------------------------------------------
# Filtered signal report
# ---------------------------------------------------------------------------

def export_filtered_signal_report(pos_signal, filtered_signal, fps,
                                  filepath, format='csv'):
    """
    Export raw POS and filtered signals.

    Columns: frame, time_s, pos_raw, pos_filtered
    """
    n = len(filtered_signal)
    # pos_signal might be longer if transient was trimmed
    pos_trimmed = pos_signal[-n:] if len(pos_signal) > n else pos_signal

    data = {
        'frame': range(1, n + 1),
        'time_s': [i / fps for i in range(n)],
        'pos_filtered': filtered_signal,
    }
    if len(pos_trimmed) == n:
        data['pos_raw'] = pos_trimmed

    df = pd.DataFrame(data)
    _save_df(df, filepath, format)
    return filepath


# ---------------------------------------------------------------------------
# RR interval report
# ---------------------------------------------------------------------------

def export_rr_report(rr_ms, rr_clean=None, rr_mask=None, filepath='', format='csv'):
    """
    Export RR intervals with optional artifact mask and interpolated values.

    Columns: beat_index, rr_raw_ms, rr_clean_ms, instantaneous_hr_bpm, artifact_rejected
    """
    n = len(rr_ms)
    inst_hr = 60000.0 / np.maximum(rr_ms, 1)

    data = {
        'beat_index': range(1, n + 1),
        'rr_raw_ms': rr_ms,
        'instantaneous_hr_bpm': np.round(inst_hr, 2),
    }
    if rr_clean is not None and len(rr_clean) == n:
        data['rr_clean_ms'] = rr_clean
        
    if rr_mask is not None and len(rr_mask) == n:
        data['artifact_interpolated'] = ~rr_mask

    df = pd.DataFrame(data)
    _save_df(df, filepath, format)
    return filepath


# ---------------------------------------------------------------------------
# HRV summary report
# ---------------------------------------------------------------------------

def export_hrv_report(hrv_metrics, session_info=None, filepath='',
                      format='csv'):
    """
    Export HRV summary metrics.

    Columns: metric, value, unit, description
    """
    rows = [
        ('mean_rr', hrv_metrics.get('mean_rr', 0), 'ms',
         'Mean RR interval'),
        ('sdnn', hrv_metrics.get('sdnn', 0), 'ms',
         'Standard deviation of NN intervals'),
        ('rmssd', hrv_metrics.get('rmssd', 0), 'ms',
         'Root mean square of successive differences'),
        ('pnn50', hrv_metrics.get('pnn50', 0), '%',
         'Percentage of successive differences > 50ms'),
        ('mean_hr', hrv_metrics.get('mean_hr', 0), 'BPM',
         'Mean heart rate from RR intervals'),
        ('min_hr', hrv_metrics.get('min_hr', 0), 'BPM',
         'Minimum instantaneous heart rate'),
        ('max_hr', hrv_metrics.get('max_hr', 0), 'BPM',
         'Maximum instantaneous heart rate'),
        ('nn_count', hrv_metrics.get('nn_count', 0), 'count',
         'Number of NN intervals analyzed'),
    ]

    if session_info:
        for k, v in session_info.items():
            rows.append((k, v, '', 'Session metadata'))

    df = pd.DataFrame(rows, columns=['metric', 'value', 'unit', 'description'])
    _save_df(df, filepath, format)
    return filepath


# ---------------------------------------------------------------------------
# Full session export
# ---------------------------------------------------------------------------

def export_full_session(session_data, output_dir, formats=None):
    """
    Export all reports for a session.

    Parameters:
        session_data: dict with keys:
            r, g, b, fps, pos_signal, filtered_signal,
            bpm, rr_ms, rr_mask, hrv, time_points, hr_values
        output_dir: directory to write reports to
        formats: list of formats ['csv', 'json', 'xlsx']

    Returns:
        dict mapping report_name → list of file paths
    """
    if formats is None:
        formats = ['csv', 'json', 'xlsx']

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    subject_info = session_data.get('subject_info', {})
    name = subject_info.get('name', '').strip()
    if not name:
        name = 'Unknown'
        
    session_dir = os.path.join(output_dir, f"{name}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    results = {}

    r = np.asarray(session_data.get('r', []))
    g = np.asarray(session_data.get('g', []))
    b = np.asarray(session_data.get('b', []))
    fps = session_data.get('fps', 30)

    for fmt in formats:
        ext = 'xlsx' if fmt == 'xlsx' else fmt

        # RGB report
        if len(r) > 0:
            path = os.path.join(session_dir, f'rgb_report_{timestamp}.{ext}')
            export_rgb_report(r, g, b, fps, path, fmt)
            results.setdefault('rgb', []).append(path)

        # Filtered signal
        pos = session_data.get('pos_signal')
        filt = session_data.get('filtered_signal')
        if filt is not None and len(filt) > 0:
            path = os.path.join(session_dir, f'signal_report_{timestamp}.{ext}')
            pos_arr = pos if pos is not None else filt
            export_filtered_signal_report(pos_arr, filt, fps, path, fmt)
            results.setdefault('signal', []).append(path)

        # RR intervals
        rr = session_data.get('rr_ms')
        rr_clean = session_data.get('rr_clean')
        if rr is not None and len(rr) > 0:
            path = os.path.join(session_dir, f'rr_report_{timestamp}.{ext}')
            rr_mask = session_data.get('rr_mask')
            export_rr_report(rr, rr_clean, rr_mask, path, fmt)
            results.setdefault('rr', []).append(path)

        # HRV summary
        hrv = session_data.get('hrv')
        if hrv:
            path = os.path.join(session_dir, f'hrv_report_{timestamp}.{ext}')
            session_info = {
                'bpm_fft': session_data.get('bpm', 0),
                'duration_frames': len(r),
                'duration_seconds': round(len(r) / fps, 2) if fps > 0 else 0,
                'fps': fps,
                'timestamp': timestamp,
                'sqi_score': hrv.get('sqi', 0.0),
                'artifact_percent': hrv.get('artifact_percent', 0.0)
            }
            # Include subject info in HRV report summary
            for k, v in subject_info.items():
                if v:
                    session_info[f'subject_{k}'] = v

            export_hrv_report(hrv, session_info, path, fmt)
            results.setdefault('hrv', []).append(path)

    # Save metadata.json
    metadata_path = os.path.join(session_dir, 'metadata.json')
    metadata = {
        'subject_info': subject_info,
        'session_timestamp': timestamp,
        'fps': fps,
        'total_frames': len(r),
        'duration_seconds': round(len(r) / fps, 2) if fps > 0 else 0,
        'bpm_fft': session_data.get('bpm', 0),
        'hrv_summary': session_data.get('hrv', {})
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    results.setdefault('metadata', []).append(metadata_path)

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_df(df, filepath, format):
    """Save a DataFrame in the specified format."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'json':
        # JSON-serializable: convert numpy types
        records = df.to_dict(orient='records')
        for rec in records:
            for k, v in rec.items():
                if isinstance(v, (np.integer,)):
                    rec[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    rec[k] = float(v)
                elif isinstance(v, (np.bool_,)):
                    rec[k] = bool(v)
        with open(filepath, 'w') as f:
            json.dump(records, f, indent=2, default=str)
    elif format == 'xlsx':
        df.to_excel(filepath, index=False, engine='openpyxl')

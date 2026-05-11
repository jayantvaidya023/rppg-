"""
realtime_camera.py — Real-time rPPG processing for webcam / IP cam.

Reuses the SAME POS + FFT pipeline from rppg_core.py.
No separate processing logic — just a rolling buffer wrapper.
"""

import time
import threading
import numpy as np
import cv2

from rppg_core import (
    get_face_cascade,
    extract_multi_roi,
    compute_windowed_signal,
    bandpass_filter,
    estimate_bpm_fft,
    MP_AVAILABLE,
    get_face_mesh
)
from preprocessing import preprocess_signal
from hrv_analysis import analyze_hrv


class CameraSource:
    """
    Unified camera source for webcam, laptop camera, and IP webcam.

    Usage:
        cam = CameraSource(0)                              # webcam
        cam = CameraSource(1)                              # second camera
        cam = CameraSource("http://192.168.1.5:8080/video") # IP webcam
    """

    def __init__(self, source=0):
        self.source = source
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera source: {self.source}"
            )
        return self

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        return self.cap.read()

    def get_fps(self):
        if self.cap is None:
            return 30.0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()


class RealtimeProcessor:
    """
    Real-time rPPG processor using a rolling buffer.

    Collects RGB values from the camera, maintains a sliding window,
    and runs the POS + FFT pipeline every update_interval seconds.

    Thread-safe for integration with the Flask web app.
    """

    def __init__(self, buffer_seconds=10, update_interval=1.0):
        self.buffer_seconds = buffer_seconds
        self.update_interval = update_interval

        # Signal buffers
        self.roi_buffers = {
            'forehead': {'r': [], 'g': [], 'b': []},
            'cheek_l': {'r': [], 'g': [], 'b': []},
            'cheek_r': {'r': [], 'g': [], 'b': []}
        }
        self.timestamps = []

        # State
        self.current_bpm = 0.0
        self.current_hrv = {}
        self.current_waveform = []
        self.current_peaks = []
        self.fps = 30.0
        self.is_running = False
        self.last_face = None
        self.face_cascade = None
        self.status = "Idle"
        self.is_file_source = False

        # Recording
        self.is_recording = False
        self.recorded_rois = {
            'forehead': {'r': [], 'g': [], 'b': []},
            'cheek_l': {'r': [], 'g': [], 'b': []},
            'cheek_r': {'r': [], 'g': [], 'b': []}
        }
        self.recorded_timestamps = []
        self.record_start_time = None

        # Threading
        self._lock = threading.Lock()
        self._camera = None
        self._thread = None
        self._stop_event = threading.Event()
        self.current_frame_jpeg = None

    def start(self, source=0):
        """Start processing from a camera source."""
        if self.is_running:
            self.stop()

        self.status = "Initializing camera..."
        self._camera = CameraSource(source)
        self._camera.open()
        self.fps = self._camera.get_fps()
        self.face_cascade = get_face_cascade()
        self.face_mesh = get_face_mesh() if MP_AVAILABLE else None
        self.is_running = True
        self._stop_event.clear()
        
        # Check if source is a file
        if isinstance(source, str) and not source.startswith("http"):
            self.is_file_source = True
        else:
            self.is_file_source = False

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop processing."""
        self._stop_event.set()
        self.is_running = False
        if self._thread is not None and threading.current_thread() != self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        elif self._thread is not None and threading.current_thread() == self._thread:
            self._thread = None
            
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def start_recording(self):
        """Begin recording session data."""
        with self._lock:
            self.is_recording = True
            for key in self.recorded_rois:
                self.recorded_rois[key] = {'r': [], 'g': [], 'b': []}
            self.recorded_timestamps = []
            self.record_start_time = time.time()

    def stop_recording(self):
        """Stop recording and return session data."""
        with self._lock:
            self.is_recording = False
            # For backward compatibility with exports, expose forehead as main rgb
            return {
                'rois': self.recorded_rois,
                'r': np.array(self.recorded_rois['forehead']['r']),
                'g': np.array(self.recorded_rois['forehead']['g']),
                'b': np.array(self.recorded_rois['forehead']['b']),
                'timestamps': list(self.recorded_timestamps),
                'fps': self.fps,
                'duration': time.time() - (self.record_start_time or time.time()),
            }

    def get_state(self):
        """Get current processor state (thread-safe)."""
        with self._lock:
            return {
                'bpm': self.current_bpm,
                'hrv': dict(self.current_hrv) if self.current_hrv else {},
                'waveform': list(self.current_waveform[-200:]),  # last 200 pts
                'is_running': self.is_running,
                'is_recording': self.is_recording,
                'buffer_size': len(self.roi_buffers['forehead']['r']),
                'fps': self.fps,
                'status': self.status,
            }

    def _capture_loop(self):
        """Main capture and processing loop (runs in background thread)."""
        last_process_time = 0
        max_buffer = int(self.buffer_seconds * self.fps)

        while not self._stop_event.is_set():
            ret, frame = self._camera.read()
            if not ret:
                if self.is_file_source:
                    self.status = "Analysis complete"
                    self.stop()
                    break
                time.sleep(0.01)
                continue

            with self._lock:
                if self.status in ["Initializing camera...", "Idle", "Analysis complete"]:
                    self.status = "Detecting face..."

            # Extract ROIs using MediaPipe (or fallback)
            rois, face_rect = extract_multi_roi(
                frame, self.face_mesh, self.face_cascade, self.last_face
            )

            # Draw ROI for the camera preview
            display_frame = frame.copy()
            if face_rect is not None:
                fx, fy, fw, fh = face_rect
                cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
                # We can draw the ROIs if we want, but FaceMesh landmarks are complex
                # We'll just keep the boxes to show we found a face
            
            ret_jpg, jpeg = cv2.imencode('.jpg', display_frame)
            if ret_jpg:
                with self._lock:
                    self.current_frame_jpeg = jpeg.tobytes()

            if rois is None:
                continue

            self.last_face = face_rect
            now = time.time()

            with self._lock:
                for key in rois:
                    self.roi_buffers[key]['r'].append(rois[key][0])
                    self.roi_buffers[key]['g'].append(rois[key][1])
                    self.roi_buffers[key]['b'].append(rois[key][2])
                self.timestamps.append(now)

                # Trim to rolling window
                if len(self.roi_buffers['forehead']['r']) > max_buffer:
                    for key in self.roi_buffers:
                        self.roi_buffers[key]['r'] = self.roi_buffers[key]['r'][-max_buffer:]
                        self.roi_buffers[key]['g'] = self.roi_buffers[key]['g'][-max_buffer:]
                        self.roi_buffers[key]['b'] = self.roi_buffers[key]['b'][-max_buffer:]
                    self.timestamps = self.timestamps[-max_buffer:]

                # Record if active
                if self.is_recording:
                    for key in rois:
                        self.recorded_rois[key]['r'].append(rois[key][0])
                        self.recorded_rois[key]['g'].append(rois[key][1])
                        self.recorded_rois[key]['b'].append(rois[key][2])
                    self.recorded_timestamps.append(now)

            # Process at update_interval
            if now - last_process_time >= self.update_interval:
                self._process_buffer()
                last_process_time = now

            # ~30 fps capture rate
            time.sleep(max(0.001, 1.0 / self.fps - 0.005))

    def generate_frames(self):
        """Generator for Flask MJPEG streaming."""
        while self.is_running:
            with self._lock:
                frame_bytes = self.current_frame_jpeg
            
            if frame_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)

    def _process_buffer(self):
        """Run POS + FFT on the current buffer (Multi-ROI & Dual Mode)."""
        from hrv_analysis import compute_sqi
        
        with self._lock:
            if len(self.roi_buffers['forehead']['r']) < 30:
                self.status = f"Collecting enough samples... ({len(self.roi_buffers['forehead']['r'])}/30)"
                return

            self.status = "Analyzing signal..."
            
            # Extract ROIs
            signals = {}
            for roi in ['forehead', 'cheek_l', 'cheek_r']:
                r = np.array(self.roi_buffers[roi]['r'])
                g = np.array(self.roi_buffers[roi]['g'])
                b = np.array(self.roi_buffers[roi]['b'])
                
                # POS Windowed
                pos = compute_windowed_signal(r, g, b, self.fps, method="pos")
                pos_filtered = bandpass_filter(pos, self.fps)
                pos_sqi = compute_sqi(pos_filtered, self.fps)
                
                # CHROM Windowed
                chrom = compute_windowed_signal(r, g, b, self.fps, method="chrom")
                chrom_filtered = bandpass_filter(chrom, self.fps)
                chrom_sqi = compute_sqi(chrom_filtered, self.fps)
                
                if chrom_sqi > pos_sqi:
                    signals[roi] = {'sig': chrom_filtered, 'sqi': chrom_sqi, 'method': 'chrom'}
                else:
                    signals[roi] = {'sig': pos_filtered, 'sqi': pos_sqi, 'method': 'pos'}

        # Fuse signals using SQI-weighted averaging
        total_sqi = sum(s['sqi'] for s in signals.values())
        if total_sqi > 1e-10:
            fused_signal = sum(s['sig'] * (s['sqi'] / total_sqi) for s in signals.values())
        else:
            fused_signal = signals['forehead']['sig']
            
        # Optional: if one is significantly better, just use it (fallback strategy)
        best_roi = max(signals.keys(), key=lambda k: signals[k]['sqi'])
        if signals[best_roi]['sqi'] > total_sqi * 0.7:  # If one ROI dominates quality
            fused_signal = signals[best_roi]['sig']

        # FFT BPM (identical to heartrate.py)
        bpm = estimate_bpm_fft(fused_signal, self.fps)

        # HRV analysis (additive)
        hrv_result = {}
        peaks = []
        try:
            hrv_data = analyze_hrv(fused_signal, self.fps)
            hrv_result = hrv_data['hrv']
            # Pass sqi and artifact_percent into hrv_result so frontend can see it
            hrv_result['sqi'] = hrv_data.get('sqi', 0.0)
            hrv_result['artifact_percent'] = hrv_data.get('artifact_percent', 0.0)
            peaks = hrv_data['peaks'].tolist()
        except Exception as e:
            print(f"HRV Error: {e}")

        with self._lock:
            self.current_bpm = bpm
            self.current_hrv = hrv_result
            self.current_waveform = fused_signal.tolist()
            self.current_peaks = peaks

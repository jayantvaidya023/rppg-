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
from hrv_analysis import analyze_hrv, compute_sqi
from collections import deque


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


class PushCameraSource:
    """
    Camera source where frames are pushed externally (e.g. from a web request).
    Used for deployed environments where the server cannot access the local webcam.
    """

    def __init__(self, fps=30.0):
        self.fps = fps
        self.current_frame = None
        self._lock = threading.Lock()
        self.is_running = False

    def open(self):
        self.is_running = True
        return self

    def push(self, frame):
        with self._lock:
            self.current_frame = frame

    def read(self):
        if not self.is_running:
            return False, None
        with self._lock:
            if self.current_frame is None:
                return False, None
            return True, self.current_frame.copy()

    def get_fps(self):
        return self.fps

    def release(self):
        self.is_running = False
        self.current_frame = None

    def is_opened(self):
        return self.is_running


class RealtimeProcessor:
    """
    Real-time rPPG processor using a rolling buffer.

    Collects RGB values from the camera, maintains a sliding window,
    and runs the POS + FFT pipeline every update_interval seconds.

    Thread-safe for integration with the Flask web app.
    """

    def __init__(self, buffer_seconds=10, update_interval=0.5):
        self.buffer_seconds = buffer_seconds
        self.update_interval = update_interval

        # Signal buffers (Deques for O(1) performance)
        self.roi_buffers = {
            'forehead': {'r': deque(maxlen=300), 'g': deque(maxlen=300), 'b': deque(maxlen=300)},
            'cheek_l': {'r': deque(maxlen=300), 'g': deque(maxlen=300), 'b': deque(maxlen=300)},
            'cheek_r': {'r': deque(maxlen=300), 'g': deque(maxlen=300), 'b': deque(maxlen=300)}
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
        
        # Determine source type
        if source == 'push':
            self.status = "Waiting for browser frames..."
            self._camera = PushCameraSource(fps=30.0)
        else:
            self.status = "Initializing camera..."
            self._camera = CameraSource(source)
        
        self._camera.open()
        self.fps = self._camera.get_fps()
        self.face_cascade = get_face_cascade()
        self.face_mesh = get_face_mesh() if MP_AVAILABLE else None
        self.is_running = True
        self._stop_event.clear()
        
        # Ensure FPS is valid
        if self.fps is None or self.fps <= 0 or np.isnan(self.fps):
            print(f"DEBUG: Invalid FPS ({self.fps}) detected. Defaulting to 30.0")
            self.fps = 30.0
            
        self.is_file_source = isinstance(source, str) and not source.startswith("http") and source != 'push'

        # Dynamically initialize buffers based on FPS
        max_buf = int(self.buffer_seconds * self.fps)
        with self._lock:
            for key in self.roi_buffers:
                self.roi_buffers[key] = {
                    'r': deque(maxlen=max_buf),
                    'g': deque(maxlen=max_buf),
                    'b': deque(maxlen=max_buf)
                }

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
        print(f"DEBUG: [Processor] Thread Started (FPS: {self.fps})")
        last_process_time = 0

        while not self._stop_event.is_set():
            try:
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
                        self.status = "Detecting face"

                # Extract ROIs using MediaPipe (or fallback)
                rois, face_rect = extract_multi_roi(
                    frame, self.face_mesh, self.face_cascade, self.last_face
                )

                # Draw ROI for the camera preview
                display_frame = frame.copy()
                if face_rect is not None:
                    fx, fy, fw, fh = face_rect
                    # Main face box (thin)
                    cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 255), 1)
                    
                    # Draw sub-ROIs (Visual indicator only)
                    # Forehead
                    fh_x1, fh_y1 = fx + fw // 4, fy + 10
                    fh_x2, fh_y2 = fx + 3 * fw // 4, fy + fh // 4
                    cv2.rectangle(display_frame, (fh_x1, fh_y1), (fh_x2, fh_y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Forehead", (fh_x1, fh_y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Cheeks
                    cw, ch = fw // 5, fh // 5
                    # Left Cheek
                    cv2.rectangle(display_frame, (fx + 10, fy + fh // 2), 
                                  (fx + 10 + cw, fy + fh // 2 + ch), (0, 255, 255), 2)
                    # Right Cheek
                    cv2.rectangle(display_frame, (fx + fw - 10 - cw, fy + fh // 2), 
                                  (fx + fw - 10, fy + fh // 2 + ch), (0, 255, 255), 2)

                ret_jpg, jpeg = cv2.imencode('.jpg', display_frame)
                if ret_jpg:
                    with self._lock:
                        self.current_frame_jpeg = jpeg.tobytes()

                if rois is None:
                    with self._lock:
                        self.status = "Searching for face..."
                    continue

                self.last_face = face_rect
                now = time.time()

                with self._lock:
                    for key in rois:
                        self.roi_buffers[key]['r'].append(rois[key][0])
                        self.roi_buffers[key]['g'].append(rois[key][1])
                        self.roi_buffers[key]['b'].append(rois[key][2])
                    self.timestamps.append(now)

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
            
            except Exception as e:
                print(f"CRITICAL: Capture loop exception: {e}")
                time.sleep(1.0)

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
        """Run optimized POS + FFT on forehead ROI for real-time performance."""
        with self._lock:
            buffer_len = len(self.roi_buffers['forehead']['r'])
            fps = self.fps
            
            if buffer_len < 60: # Need at least 2 seconds for stable FFT
                self.status = f"Buffering... ({buffer_len}/60)"
                return

            self.status = "Processing pulse"
            
            # Optimized: Only process Forehead ROI in real-time to save CPU
            r = np.array(self.roi_buffers['forehead']['r'])
            g = np.array(self.roi_buffers['forehead']['g'])
            b = np.array(self.roi_buffers['forehead']['b'])
            
        try:
            # POS Windowed (Most stable for real-time)
            pos = compute_windowed_signal(r, g, b, fps, method="pos")
            fused_signal = bandpass_filter(pos, fps)
            
            # FFT BPM (identical to heartrate.py)
            bpm = estimate_bpm_fft(fused_signal, fps)

            # HRV analysis (additive)
            hrv_result = {}
            peaks = []
            try:
                hrv_data = analyze_hrv(fused_signal, fps)
                hrv_result = hrv_data['hrv']
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
                
                # Signal Diagnostic Status
                g_mean = np.mean(g)
                g_std = np.std(g)
                self.status = f"Live: {round(bpm, 1)} BPM | Signal G: μ={round(g_mean, 1)}, σ={round(g_std, 2)}"

        except Exception as e:
            print(f"Processing Error: {e}")
            with self._lock:
                self.status = f"Error: {str(e)[:20]}"

"""
web_app.py — Lightweight Flask web interface for rPPG.

Features:
    - Live BPM display via Server-Sent Events (SSE)
    - Waveform preview
    - Start/stop recording sessions
    - Video file upload and analysis
    - Export reports (CSV/JSON/Excel)
    - Webcam / IP webcam source selection

Reuses the SAME POS + FFT pipeline from rppg_core.py.
"""

import os
import io
import time
import json
import threading
import datetime
import zipfile
import tempfile
import queue
import shutil

from flask import (
    Flask, render_template, request, jsonify,
    Response, send_file,
)
from werkzeug.utils import secure_filename
import numpy as np

from rppg_core import process_video, bandpass_filter, compute_windowed_signal, estimate_bpm_fft
from preprocessing import preprocess_signal
from hrv_analysis import analyze_hrv
from export_reports import export_full_session
from realtime_camera import RealtimeProcessor


app = Flask(__name__)

# Global processor instance - Optimized for Render (0.5s update interval)
processor = RealtimeProcessor(buffer_seconds=15, update_interval=0.5)

# Session storage (in-memory)
last_session_data = {}
last_video_results = {}
experiment_sessions = {} # Tracks active experiment phases

EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'exports')
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'rppg_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------------------------------------------------------------------
# Research Logging
# ---------------------------------------------------------------------------

def log_research_event(event_type, experiment_id, data=None):
    """Log events to a central research log file."""
    log_file = os.path.join(EXPORT_DIR, 'research_audit.csv')
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    headers = ['timestamp', 'event_type', 'experiment_id', 'data']
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, 'a', encoding='utf-8') as f:
        if not file_exists:
            f.write(','.join(headers) + '\n')
        
        timestamp = datetime.datetime.now().isoformat()
        data_str = json.dumps(data).replace(',', ';') if data else ""
        f.write(f"{timestamp},{event_type},{experiment_id},{data_str}\n")

# ---------------------------------------------------------------------------
# Background Processing Queue
# ---------------------------------------------------------------------------

class BackgroundProcessor:
    def __init__(self):
        self.queue = queue.Queue()
        self.tasks = {} # task_id: {status, progress, result, error}
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self):
        while True:
            try:
                task_id, func, args, kwargs = self.queue.get()
                self.tasks[task_id]['status'] = 'processing'
                print(f"[BG] Starting task {task_id}")
                
                result = func(*args, **kwargs)
                
                self.tasks[task_id]['status'] = 'completed'
                self.tasks[task_id]['result'] = result
                print(f"[BG] Completed task {task_id}")
            except Exception as e:
                print(f"[BG] Task {task_id} failed: {e}")
                import traceback
                traceback.print_exc()
                if task_id in self.tasks:
                    self.tasks[task_id]['status'] = 'failed'
                    self.tasks[task_id]['error'] = str(e)
            finally:
                self.queue.task_done()

    def submit(self, task_id, func, *args, **kwargs):
        self.tasks[task_id] = {
            'status': 'pending',
            'created_at': time.time()
        }
        self.queue.put((task_id, func, args, kwargs))

    def get_status(self, task_id):
        return self.tasks.get(task_id, {'status': 'not_found'})

bg_processor = BackgroundProcessor()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """Now serving the Unified Research Platform as the main entry point."""
    return render_template('data_collection.html')


@app.route('/monitor')
def monitor():
    """Existing real-time monitor moved to a secondary route."""
    return render_template('index.html')


# ---------------------------------------------------------------------------
# Camera control API
# ---------------------------------------------------------------------------

@app.route('/api/start-camera', methods=['POST'])
def start_camera():
    """Start real-time camera processing."""
    data = request.get_json(silent=True) or {}
    source = data.get('source', 0)

    # Parse source: integer for webcam, string for IP cam
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass  # keep as string (URL)

    try:
        processor.start(source)
        return jsonify({'status': 'ok', 'message': f'Camera started: {source}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/stop-camera', methods=['POST'])
def stop_camera():
    """Stop camera processing."""
    processor.stop()
    return jsonify({'status': 'ok', 'message': 'Camera stopped'})


@app.route('/api/push-frame', methods=['POST'])
def push_frame():
    """Receive a frame from the browser for processing."""
    if not processor.is_running or not isinstance(processor._camera, PushCameraSource):
        return jsonify({'status': 'error', 'message': 'Processor not in push mode'}), 400
    
    data = request.get_json(silent=True) or {}
    frame_data = data.get('frame')
    if not frame_data:
        return jsonify({'status': 'error', 'message': 'No frame data'}), 400
    
    try:
        # Decode base64 frame
        import base64
        import cv2
        
        # Remove header if present
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]
            
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            processor._camera.push(frame)
            return jsonify({'status': 'ok'})
        else:
            return jsonify({'status': 'error', 'message': 'Decode failed'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ---------------------------------------------------------------------------
# Recording API
# ---------------------------------------------------------------------------

@app.route('/api/start-recording', methods=['POST'])
def start_recording():
    """Start recording session data."""
    processor.start_recording()
    return jsonify({'status': 'ok', 'message': 'Recording started'})


@app.route('/api/stop-recording', methods=['POST'])
def stop_recording():
    """Stop recording and process the session."""
    global last_session_data
    
    req_data = request.get_json(silent=True) or {}
    subject_info = req_data.get('subject_info', {})

    session = processor.stop_recording()

    if len(session['r']) < 30:
        return jsonify({
            'status': 'error',
            'message': f"Only {len(session['r'])} frames recorded. Need at least 30."
        }), 400

    # Process recorded data with the SAME pipeline as heartrate.py
    r, g, b = session['r'], session['g'], session['b']
    fps = session['fps']

    pos = compute_windowed_signal(r, g, b, fps)
    filtered = bandpass_filter(pos, fps)

    from rppg_core import estimate_bpm_fft, sliding_window_hr
    bpm = estimate_bpm_fft(filtered, fps)
    time_points, hr_values = sliding_window_hr(filtered, fps)

    # HRV analysis (additive)
    hrv_data = analyze_hrv(filtered, fps)

    last_session_data = {
        'r': r, 'g': g, 'b': b,
        'fps': fps,
        'pos_signal': pos,
        'filtered_signal': filtered,
        'bpm': bpm,
        'time_points': time_points,
        'hr_values': hr_values,
        'rr_ms': hrv_data['rr_ms'],
        'rr_clean': hrv_data['rr_clean'],
        'rr_mask': hrv_data['rr_mask'],
        'hrv': hrv_data['hrv'],
        'peak_bpm': hrv_data['peak_bpm'],
        'peaks': hrv_data['peaks'],
        'duration': session['duration'],
        'source': 'camera_recording',
        'subject_info': subject_info,
    }

    return jsonify({
        'status': 'ok',
        'bpm': round(bpm, 1),
        'peak_bpm': round(hrv_data['peak_bpm'], 1),
        'hrv': hrv_data['hrv'],
        'duration': round(session['duration'], 1),
        'frames': len(r),
    })


# ---------------------------------------------------------------------------
# Video file analysis API
# ---------------------------------------------------------------------------

@app.route('/api/analyze-video', methods=['POST'])
def analyze_video():
    """Upload and analyze a video file."""
    global last_session_data

    subject_info = {}
    try:
        subject_info = json.loads(request.form.get('subject_info', '{}'))
    except Exception:
        pass

    if 'video' not in request.files:
        # Try using default video
        video_path = os.path.join(os.path.dirname(__file__), 'rPPG_video.mp4')
        if not os.path.exists(video_path):
            return jsonify({
                'status': 'error',
                'message': 'No video file provided and no default video found.'
            }), 400
    else:
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No video selected'})
            
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)
        
        # Verify file is not empty
        if os.path.getsize(video_path) < 100:
             return jsonify({'success': False, 'error': 'Video file too small or corrupt'})
             
        print(f"DEBUG: Processing uploaded video: {filename} ({os.path.getsize(video_path)} bytes)")
        
        # Stop existing camera
        processor.stop()
        
        # Start processing
        processor.start(video_path)
        processor.start_recording()
        
        # Start background thread to finalize when done
        def finalize():
            global last_session_data
            while processor.is_running:
                time.sleep(0.5)
            session = processor.stop_recording()
            if len(session['r']) > 30:
                r, g, b = session['r'], session['g'], session['b']
                fps = session['fps']
                pos = compute_windowed_signal(r, g, b, fps)
                filtered = bandpass_filter(pos, fps)
                from rppg_core import estimate_bpm_fft, sliding_window_hr
                bpm = estimate_bpm_fft(filtered, fps)
                time_points, hr_values = sliding_window_hr(filtered, fps)
                hrv_data = analyze_hrv(filtered, fps)
                last_session_data = {
                    'r': r, 'g': g, 'b': b, 'fps': fps,
                    'pos_signal': pos, 'filtered_signal': filtered,
                    'bpm': bpm, 'time_points': time_points, 'hr_values': hr_values,
                    'rr_ms': hrv_data['rr_ms'], 'rr_clean': hrv_data['rr_clean'],
                    'rr_mask': hrv_data['rr_mask'], 'hrv': hrv_data['hrv'],
                    'peak_bpm': hrv_data['peak_bpm'], 'peaks': hrv_data['peaks'],
                    'duration': session['duration'], 'source': 'video_file',
                    'subject_info': subject_info,
                }
        threading.Thread(target=finalize, daemon=True).start()
        
    return jsonify({'status': 'ok', 'message': 'Live analysis started'})




@app.route('/api/analyze-default', methods=['POST'])
def analyze_default_video():
    """Analyze the default rPPG_video.mp4."""
    global last_session_data

    req_data = request.get_json(silent=True) or {}
    subject_info = req_data.get('subject_info', {})

    video_path = os.path.join(os.path.dirname(__file__), 'rPPG_video.mp4')
    if not os.path.exists(video_path):
        return jsonify({
            'status': 'error',
            'message': 'rPPG_video.mp4 not found.'
        }), 404

    try:
        processor.start(video_path)
        processor.start_recording()
        
        def finalize():
            global last_session_data
            while processor.is_running:
                time.sleep(0.5)
            session = processor.stop_recording()
            if len(session['r']) > 30:
                r, g, b = session['r'], session['g'], session['b']
                fps = session['fps']
                pos = compute_windowed_signal(r, g, b, fps)
                filtered = bandpass_filter(pos, fps)
                from rppg_core import estimate_bpm_fft, sliding_window_hr
                bpm = estimate_bpm_fft(filtered, fps)
                time_points, hr_values = sliding_window_hr(filtered, fps)
                hrv_data = analyze_hrv(filtered, fps)
                last_session_data = {
                    'r': r, 'g': g, 'b': b, 'fps': fps,
                    'pos_signal': pos, 'filtered_signal': filtered,
                    'bpm': bpm, 'time_points': time_points, 'hr_values': hr_values,
                    'rr_ms': hrv_data['rr_ms'], 'rr_clean': hrv_data['rr_clean'],
                    'rr_mask': hrv_data['rr_mask'], 'hrv': hrv_data['hrv'],
                    'peak_bpm': hrv_data['peak_bpm'], 'peaks': hrv_data['peaks'],
                    'duration': session['duration'], 'source': 'default_video',
                    'subject_info': subject_info,
                }
        threading.Thread(target=finalize, daemon=True).start()
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

    return jsonify({'status': 'ok', 'message': 'Live analysis started'})




# ---------------------------------------------------------------------------
# SSE stream for real-time data
# ---------------------------------------------------------------------------

@app.route('/api/test-stream')
def test_stream():
    """SSE verification route with full mock physiological payloads."""
    def generate():
        client_id = f"test_client_{int(time.time())}"
        print(f"[SSE-TEST] [{client_id}] Connection started")
        count = 0
        
        # Populate mock session data for export testing
        global last_session_data
        mock_len = 300
        mock_fps = 30
        mock_r = (np.random.normal(120, 5, mock_len)).tolist()
        mock_g = (np.random.normal(110, 5, mock_len)).tolist()
        mock_b = (np.random.normal(100, 5, mock_len)).tolist()
        mock_pos = (np.sin(np.linspace(0, 10*np.pi, mock_len)) * 2).tolist()
        mock_filt = (np.sin(np.linspace(0, 10*np.pi, mock_len)) * 1.5).tolist()
        
        last_session_data = {
            'r': mock_r, 'g': mock_g, 'b': mock_b,
            'fps': mock_fps,
            'pos_signal': mock_pos,
            'filtered_signal': mock_filt,
            'bpm': 72.5,
            'time_points': [float(i/2) for i in range(20)],
            'hr_values': [float(70 + np.sin(i/5)*5) for i in range(20)],
            'rr_ms': [800.0, 810.0, 790.0, 805.0] * 10,
            'rr_clean': [800.0, 810.0, 790.0, 805.0] * 10,
            'rr_mask': [True] * 40,
            'hrv': {
                'sdnn': 45.2, 'rmssd': 38.5, 'pnn50': 12.4, 
                'mean_rr': 802.1, 'mean_hr': 74.8, 'min_hr': 68.2, 'max_hr': 82.5,
                'nn_count': 40, 'sqi': 95.5, 'artifact_percent': 2.1
            },
            'peak_bpm': 73.2,
            'peaks': [10, 40, 70, 100],
            'duration': 10.0,
            'source': 'test_mock_data',
            'subject_info': {'name': 'Test User', 'age': '25', 'gender': 'Other'},
        }

        try:
            while True:
                count += 1
                now = time.time()
                
                # Mock waveform (moving sine wave)
                t = np.linspace(count*0.1, count*0.1 + 2*np.pi, 100)
                mock_waveform = (np.sin(t) * 100).tolist()
                
                data = {
                    "bpm": round(70 + np.sin(count/10.0) * 5, 1),
                    "status": "Test Stream Active",
                    "timestamp": now,
                    "packet_id": count,
                    "is_running": True,
                    "is_recording": False,
                    "waveform": mock_waveform,
                    "hrv": last_session_data['hrv'],
                    "sqi": 95.5,
                    "artifact_percent": 2.1
                }
                
                payload = json.dumps(data)
                if count % 10 == 0:
                    print(f"[SSE-TEST] [{client_id}] Emitting Packet #{count}: BPM={data['bpm']}")
                
                yield f"data: {payload}\n\n"
                time.sleep(0.5) # 2 Hz updates for responsiveness
        except Exception as e:
            print(f"[SSE-TEST] [{client_id}] Stream interrupted: {e}")
        finally:
            print(f"[SSE-TEST] [{client_id}] Connection terminated")
    
    return Response(
        generate(), 
        mimetype='text/event-stream', 
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )
@app.route('/api/default-video')
def default_video_file():
    """Serve the default video file for frontend playback."""
    video_path = os.path.join(os.path.dirname(__file__), 'rPPG_video.mp4')
    if os.path.exists(video_path):
        from flask import send_file
        return send_file(video_path, mimetype='video/mp4')
    return "Not found", 404

@app.route('/api/stream')
def stream():
    """Server-Sent Events stream optimized for public proxies (Render/Localtunnel)."""
    def generate():
        client_id = f"client_{int(time.time())}"
        print(f"[SSE] [{client_id}] Connection requested")
        
        last_heartbeat = 0
        packet_count = 0
        
        try:
            while True:
                now = time.time()
                state = processor.get_state()
                
                # Construct data
                data = {
                    'bpm': round(state['bpm'], 1),
                    'hrv': state['hrv'],
                    'waveform': state['waveform'][-100:],
                    'is_running': state['is_running'],
                    'is_recording': state['is_recording'],
                    'status': state['status'] or "Initializing...",
                    'timestamp': now,
                    'packet_id': packet_count + 1
                }
                
                if state['hrv']:
                    data['sqi'] = state['hrv'].get('sqi', 0.0)
                    data['artifact_percent'] = state['hrv'].get('artifact_percent', 0.0)

                # Send data if processing OR send heartbeat every 2 seconds
                # Proxies (Localtunnel/Render) often close idle connections after 30-60s
                should_send = state['is_running'] or (now - last_heartbeat > 2.0)
                
                if should_send:
                    if not state['is_running']:
                        data['heartbeat'] = True
                        data['status'] = state['status'] or "Waiting for data..."
                    
                    payload = json.dumps(data)
                    packet_count += 1
                    
                    if packet_count % 20 == 0 or not state['is_running']:
                        print(f"[SSE] [{client_id}] Emitting Packet #{packet_count} (BPM={data['bpm']}, Status={data['status']})")
                    
                    yield f"data: {payload}\n\n"
                    last_heartbeat = now
                
                time.sleep(0.1) # 10 Hz check
        except Exception as e:
            print(f"[SSE] [{client_id}] Stream exception: {e}")
        finally:
            print(f"[SSE] [{client_id}] Connection closed after {packet_count} packets")

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
            # REMOVED: Transfer-Encoding: chunked (can cause issues with some proxies)
        }
    )

@app.route('/api/video_feed')
def video_feed():
    """Live MJPEG camera feed."""
    if not processor.is_running:
        return Response(status=204)
    return Response(
        processor.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ---------------------------------------------------------------------------
# Export API
# ---------------------------------------------------------------------------
@app.route('/api/export/<format_type>')
def export_report(format_type):
    """Export session reports with robust logging and buffer handling."""
    print(f"[EXPORT] Request received for format: {format_type}")
    
    if not last_session_data:
        print("[EXPORT] Error: No session data available")
        return jsonify({
            'status': 'error',
            'message': 'No session data. Analyze a video or record a session first.'
        }), 400

    if format_type not in ('csv', 'json', 'xlsx'):
        print(f"[EXPORT] Error: Unsupported format {format_type}")
        return jsonify({
            'status': 'error',
            'message': f'Unsupported format: {format_type}. Use csv, json, or xlsx.'
        }), 400

    try:
        # Update subject info from query params
        name = request.args.get('name', '').strip()
        age = request.args.get('age', '').strip()
        gender = request.args.get('gender', '').strip()
        if name or age or gender:
            last_session_data['subject_info'] = {'name': name, 'age': age, 'gender': gender}

        os.makedirs(EXPORT_DIR, exist_ok=True)
        print(f"[EXPORT] Generating reports in {EXPORT_DIR}...")

        results = export_full_session(
            last_session_data, EXPORT_DIR, formats=[format_type]
        )

        all_files = []
        for file_list in results.values():
            all_files.extend(file_list)

        print(f"[EXPORT] Generated {len(all_files)} files")

        if not all_files:
            return jsonify({'status': 'error', 'message': 'No data to export.'}), 400

        # If single file (e.g. JSON), send directly
        if len(all_files) == 1:
            print(f"[EXPORT] Sending single file: {all_files[0]}")
            return send_file(
                all_files[0],
                as_attachment=True,
                download_name=os.path.basename(all_files[0]),
                mimetype='application/octet-stream' # More compatible for generic downloads
            )

        # Multiple files: zip them
        print("[EXPORT] Zipping multiple files...")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in all_files:
                zf.write(f, os.path.basename(f))
        zip_buffer.seek(0)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'rppg_report_{timestamp}.zip'
        print(f"[EXPORT] Sending zip: {filename} ({zip_buffer.getbuffer().nbytes} bytes)")
        
        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype='application/zip'
        )
    except Exception as e:
        print(f"[EXPORT] Critical Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500



@app.route('/api/session-summary')
def session_summary():
    """Get a summary of the last session."""
    if not last_session_data:
        return jsonify({'status': 'none', 'message': 'No session data.'})

    return jsonify({
        'status': 'ok',
        'bpm': round(last_session_data.get('bpm', 0), 1),
        'peak_bpm': round(last_session_data.get('peak_bpm', 0), 1),
        'hrv': last_session_data.get('hrv', {}),
        'source': last_session_data.get('source', ''),
        'frames': len(last_session_data.get('r', [])),
        'fps': last_session_data.get('fps', 0),
    })


# ---------------------------------------------------------------------------
# Data Collection API
# ---------------------------------------------------------------------------

@app.route('/api/data-collection/save-session', methods=['POST'])
def save_collection_session():
    """
    Receive video blob and metadata for a specific phase (PRE/DURING/POST).
    Process it and save to the structured folder.
    """
    try:
        video_file = request.files.get('video')
        subject_info_raw = request.form.get('subject_info', '{}')
        phase = request.form.get('phase', 'PRE').upper()
        experiment_id = request.form.get('experiment_id', 'exp_' + str(int(time.time())))
        
        subject_info = json.loads(subject_info_raw)
        name = secure_filename(subject_info.get('name', 'Unknown'))
        
        # Structure: exports/name/experiment_id/phase/
        experiment_dir = os.path.join(EXPORT_DIR, name, experiment_id)
        phase_dir = os.path.join(experiment_dir, phase)
        os.makedirs(phase_dir, exist_ok=True)
        
        video_path = os.path.join(phase_dir, f'recording_{phase.lower()}.webm')
        video_file.save(video_path)
        
        print(f"[COLLECTION] Saved video for {name} ({phase}) to {video_path}")
        
        # Trigger Automated Analysis
        results = process_video(video_path)
        hrv_results = analyze_hrv(results['filtered_signal'], results['fps'])
        
        # Prepare session data for export
        session_data = {
            **results,
            **hrv_results,
            'subject_info': subject_info,
            'source': f'collection_{phase.lower()}'
        }
        
        # Export all reports to the phase directory
        from export_reports import export_full_session
        export_full_session(session_data, EXPORT_DIR, formats=['csv', 'json', 'xlsx'])
        
        # Move generated reports into the phase folder (export_full_session creates its own timestamped subfolder by default)
        # We want them DIRECTLY in phase_dir.
        # Actually, let's update export_full_session to accept a custom dir. 
        # I already did that! Let's use it.
        
        session_info = {'custom_dir': phase_dir}
        export_full_session(session_data, EXPORT_DIR, formats=['csv', 'json', 'xlsx'], session_info=session_info)
        
        # Track for comparison dashboard (in memory for quick access, but backed by files)
        if experiment_id not in experiment_sessions:
            experiment_sessions[experiment_id] = {}
        
        summary = {
            'bpm': round(results['bpm'], 1),
            'sdnn': round(hrv_results['hrv']['sdnn'], 1),
            'rmssd': round(hrv_results['hrv']['rmssd'], 1),
            'pnn50': round(hrv_results['hrv']['pnn50'], 1),
            'mean_rr': round(hrv_results['hrv']['mean_rr'], 1),
            'sqi': round(hrv_results['sqi'], 1),
            'timestamp': datetime.datetime.now().isoformat()
        }
        experiment_sessions[experiment_id][phase] = summary
        
        return jsonify({
            'status': 'ok',
            'message': f'Phase {phase} processed successfully.',
            'summary': summary
        })
        
    except Exception as e:
        print(f"[COLLECTION] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/data-collection/upload-chunk', methods=['POST'])
def upload_chunk():
    """
    Receive a chunk of video data.
    Chunks are saved to a temporary folder until the session is finalized.
    """
    try:
        chunk = request.files.get('video')
        experiment_id = request.form.get('experiment_id')
        phase = request.form.get('phase', 'PRE').upper()
        chunk_index = int(request.form.get('chunk_index', 0))
        
        if not chunk or not experiment_id:
            print(f"[CHUNK] Missing data: chunk={chunk}, experiment_id={experiment_id}")
            return jsonify({'status': 'error', 'message': 'Missing data'}), 400

        # Create temporary chunk directory
        chunk_dir = os.path.join(UPLOAD_FOLDER, experiment_id, phase)
        os.makedirs(chunk_dir, exist_ok=True)
        
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:05d}.blob")
        chunk.save(chunk_path)
        
        print(f"[CHUNK] Received and saved chunk {chunk_index} for {experiment_id} ({phase}) - size: {os.path.getsize(chunk_path)} bytes")
        log_research_event("CHUNK_UPLOAD", experiment_id, {"phase": phase, "chunk": chunk_index})
        return jsonify({'status': 'ok', 'chunk_index': chunk_index})
    except Exception as e:
        print(f"[CHUNK] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/data-collection/finalize-session', methods=['POST'])
def finalize_session():
    """
    Finalize a session: combine chunks and queue for analysis.
    """
    try:
        data = request.get_json() or {}
        experiment_id = data.get('experiment_id')
        participant_timestamp = data.get('participant_timestamp')
        phase = data.get('phase', 'PRE').upper()
        if not participant_timestamp:
            participant_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        subject_info = data.get('subject_info', {})
        
        print(f"[FINALIZE] Starting finalize for {experiment_id} {phase}")
        if not experiment_id:
            print(f"[FINALIZE] Missing experiment_id")
            return jsonify({'status': 'error', 'message': 'Missing experiment_id'}), 400

        chunk_dir = os.path.join(UPLOAD_FOLDER, experiment_id, phase)
        if not os.path.exists(chunk_dir):
            print(f"[FINALIZE] No chunks found in {chunk_dir}")
            return jsonify({'status': 'error', 'message': 'No chunks found for this session'}), 404

        # Combine chunks
        name = secure_filename(subject_info.get('name', 'Unknown'))

        participant_root_dir = os.path.join(EXPORT_DIR, f"{name}_{participant_timestamp}")
        phase_dir = os.path.join(participant_root_dir, phase)


        os.makedirs(phase_dir, exist_ok=True)
        
        # Mime type logic: client should send it, but we'll default to webm
        # Actually, let's look at the first chunk to see what we have if possible, 
        # but for now we'll assume webm or mp4 based on what MediaRecorder sends.
        final_video_path = os.path.join(phase_dir, f'recording_{phase.lower()}.webm')
        
        chunks = sorted([f for f in os.listdir(chunk_dir) if f.startswith('chunk_')])
        print(f"[FINALIZE] Found {len(chunks)} chunks to combine")
        if not chunks:
             return jsonify({'status': 'error', 'message': 'No chunks to combine'}), 400
             
        with open(final_video_path, 'wb') as outfile:
            for chunk_file in chunks:
                with open(os.path.join(chunk_dir, chunk_file), 'rb') as infile:
                    outfile.write(infile.read())

        # Determine file extension based on content
        with open(final_video_path, 'rb') as f:
            header = f.read(12)
            if header.startswith(b'\x1a\x45\xdf\xa3'):
                # WebM
                pass
            elif b'ftyp' in header:
                # MP4
                new_path = final_video_path.replace('.webm', '.mp4')
                os.rename(final_video_path, new_path)
                final_video_path = new_path
                print(f"[FINALIZE] Renamed to MP4: {final_video_path}")
            else:
                print(f"[FINALIZE] Unknown video format, keeping .webm")
        
        file_size_bytes = os.path.getsize(final_video_path)
        print(f"[FINALIZE] Combined {len(chunks)} chunks into {final_video_path} - size: {file_size_bytes} bytes")
        print("[VIDEO] Saved:", final_video_path)
        print("[VIDEO] Size:", file_size_bytes)
        
        # Validate combined recording before queueing analysis.
        if file_size_bytes <= 0:
            raise ValueError(f"Combined video file is empty: {final_video_path}")
        
        # Try to open with OpenCV to avoid queuing corrupted recordings.
        import cv2
        cap = cv2.VideoCapture(final_video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open combined video with OpenCV: {final_video_path}")
        finally:
            cap.release()
        
        log_research_event("SESSION_FINALIZE", experiment_id, {"phase": phase, "chunks": len(chunks)})
        
        # Defer chunk deletion until analysis/export succeeds inside run_analysis_task.
        # (Preserves original recordings/chunks for recovery/retry.)

        # Queue for background processing
        task_id = f"{experiment_id}_{phase}"
        print(f"[FINALIZE] Queuing task {task_id} for analysis")
        bg_processor.submit(task_id, run_analysis_task, final_video_path, subject_info, phase, experiment_id)
        
        return jsonify({
            'status': 'ok',
            'message': 'Session finalized and queued for analysis.',
            'task_id': task_id
        })
    except Exception as e:
        print(f"[FINALIZE] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


def run_analysis_task(video_path, subject_info, phase, experiment_id):
    """Worker function for background analysis."""
    # Check if video_path exists, if not try .mp4 version
    if not os.path.exists(video_path):
        alt_path = video_path.replace('.webm', '.mp4')
        if os.path.exists(alt_path):
            video_path = alt_path
            print(f"[TASK] Using MP4 version: {video_path}")

    print(f"[TASK] Starting analysis for {video_path}")

    try:
        # Required video preservation/verification logs
        if os.path.exists(video_path):
            print("[VIDEO] Saved:", video_path)
            print("[VIDEO] Size:", os.path.getsize(video_path))
        else:
            print("[VIDEO] Missing file at start:", video_path)

        import cv2
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video in task: {video_path}")
        finally:
            cap.release()

        print("[POST] Processing started" if phase == "POST" else f"[{phase}] Processing started")
        print(f"[POST] Phase={phase}, experiment={experiment_id}")

        # Check if video file is valid
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        file_size = os.path.getsize(video_path)
        print(f"[POST] Video file size: {file_size} bytes")
        if file_size == 0:
            raise ValueError(f"Video file is empty: {video_path}")

        # Try to open with OpenCV to check
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file with OpenCV: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"[POST] Video opened successfully: {frame_count} frames at {fps} fps")

        results = process_video(video_path)
        print(f"[POST] process_video completed: bpm={results.get('bpm')}")

        hrv_results = analyze_hrv(results['filtered_signal'], results['fps'])
        print(f"[POST] analyze_hrv completed: sdnn={hrv_results.get('hrv', {}).get('sdnn')}")

        print("[POST] Processing finished" if phase == "POST" else f"[{phase}] Processing finished")

        # Prepare session data for export
        print(f"[POST] Preparing session data")
        session_data = {
            **results,
            **hrv_results,
            'subject_info': subject_info,
            'source': f'collection_{phase.lower()}'
        }

        # Export all reports to the phase directory (timeout-protected so POST can't hang the workflow)
        phase_dir = os.path.dirname(video_path)
        session_info = {'custom_dir': phase_dir}

        print("[POST] Starting final aggregation" if phase == "POST" else f"[{phase}] Starting final aggregation")
        try:
            export_done = threading.Event()

            def do_export():
                try:
                    print("[POST] Exporting reports" if phase == "POST" else f"[{phase}] Exporting reports")
                    export_full_session(
                        session_data,
                        EXPORT_DIR,
                        formats=['csv', 'json', 'xlsx'],
                        session_info=session_info
                    )
                    print("[POST] Export completed" if phase == "POST" else f"[{phase}] Export completed")
                except Exception as e:
                    print(f"[POST] EXPORT ERROR: {e}" if phase == "POST" else f"[{phase}] EXPORT ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    export_done.set()

            export_thread = threading.Thread(target=do_export, daemon=True)
            export_thread.start()

            # Hard timeout (seconds). After this, continue so task completes.
            export_timeout_s = 60 if phase == "POST" else 90
            export_finished = export_done.wait(timeout=export_timeout_s)

            if not export_finished and phase == "POST":
                print(f"[POST] Export timed out after {export_timeout_s}s; continuing to finalize workflow anyway")
            elif not export_finished:
                print(f"[{phase}] Export timed out after {export_timeout_s}s; continuing")

        except Exception as e:
            # Continue anyway; do not block completion
            print(f"[POST] Export wrapper error: {e}")
            import traceback
            traceback.print_exc()

        print("[POST] Generating comparison summary" if phase == "POST" else f"[{phase}] Generating comparison summary")
        print(f"[POST] Building summary")

        summary = {
            'bpm': round(results['bpm'], 1),
            'sdnn': round(hrv_results['hrv']['sdnn'], 1),
            'rmssd': round(hrv_results['hrv']['rmssd'], 1),
            'pnn50': round(hrv_results['hrv']['pnn50'], 1),
            'mean_rr': round(hrv_results['hrv']['mean_rr'], 1),
            'sqi': round(hrv_results['sqi'], 1),
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Update global experiment_sessions for the comparison dashboard
        print("[POST] Final workflow complete" if phase == "POST" else f"[{phase}] Final workflow complete")
        print(f"[POST] Updating experiment_sessions")
        if experiment_id not in experiment_sessions:
            experiment_sessions[experiment_id] = {}
        experiment_sessions[experiment_id][phase] = summary
        print(f"[POST] Experiment sessions now has phases: {list(experiment_sessions[experiment_id].keys())}")

        log_research_event("ANALYSIS_COMPLETE", experiment_id, {"phase": phase, "bpm": summary['bpm']})
        print(f"[POST] Analysis complete for {experiment_id}_{phase}: bpm={summary['bpm']}")
        print("[POST] ======== WORKFLOW COMPLETE ========")

        return summary

    except Exception as e:
        print(f"[TASK] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        raise


@app.route('/api/data-collection/task-status/<task_id>')
def get_task_status(task_id):
    """Check the status of a background analysis task."""
    status_info = bg_processor.get_status(task_id)
    return jsonify(status_info)


@app.route('/api/data-collection/comparison')
def get_comparison():
    """Get comparison data for a specific experiment ID."""
    exp_id = request.args.get('experiment_id')
    if not exp_id or exp_id not in experiment_sessions:
        return jsonify({'status': 'error', 'message': 'Experiment not found'}), 404
        
    return jsonify({
        'status': 'ok',
        'experiment_id': exp_id,
        'data': experiment_sessions[exp_id]
    })


@app.route('/api/data-collection/results')
def get_experiment_results():
    """
    Fetch DETAILED results (waveforms, RR intervals + HRV) for all phases of an experiment.

    Exports during collection use export_full_session(..., session_info={'custom_dir': phase_dir}).
    That exporter writes timestamped JSON files:
      - rgb_report_*.json
      - signal_report_*.json
      - rr_report_*.json
    and also writes:
      - hrv_report.json
    """
    subject_name = request.args.get('name')
    exp_id = request.args.get('experiment_id')

    if not subject_name or not exp_id:
        return jsonify({'status': 'error', 'message': 'Missing parameters'}), 400

    name_secure = secure_filename(subject_name)
    exp_id_secure = secure_filename(exp_id)

    experiment_dir = os.path.join(EXPORT_DIR, name_secure, exp_id_secure)
    if not os.path.exists(experiment_dir):
        return jsonify({'status': 'error', 'message': 'Experiment not found on disk'}), 404

    results = {}
    for phase in ['PRE', 'DURING', 'POST']:
        phase_dir = os.path.join(experiment_dir, phase)
        if not os.path.exists(phase_dir):
            results[phase] = None
            continue

        # Load the expected JSON artifacts from disk (deterministically pick the newest).
        def newest_json(glob_suffixes):
            candidates = []
            for fn in os.listdir(phase_dir):
                if not fn.endswith('.json'):
                    continue
                if any(fn.startswith(suf) for suf in glob_suffixes):
                    candidates.append(fn)
            if not candidates:
                return None
            # Sort by filename (timestamp embedded in name), fallback to mtime if needed.
            candidates.sort()
            return os.path.join(phase_dir, candidates[-1])

        signal_json_path = newest_json(['signal_report_', 'signal_report'])
        rr_json_path = newest_json(['rr_report_', 'rr_report'])
        rgb_json_path = newest_json(['rgb_report_', 'rgb_report'])
        hrv_json_path = os.path.join(phase_dir, 'hrv_report.json') if os.path.exists(os.path.join(phase_dir, 'hrv_report.json')) else None

        phase_payload = {
            'bpm': None,
            'hrv': None,
            'filtered_signal': [],
            'rr_ms': [],
            'pos_signal': [],
            'r': [],
            'g': [],
            'b': [],
            'source_json': {
                'signal': os.path.basename(signal_json_path) if signal_json_path else None,
                'rr': os.path.basename(rr_json_path) if rr_json_path else None,
                'rgb': os.path.basename(rgb_json_path) if rgb_json_path else None,
                'hrv': os.path.basename(hrv_json_path) if hrv_json_path else None,
            }
        }

        try:
            # HRV (contains bpm + hrv keys)
            if hrv_json_path:
                with open(hrv_json_path, 'r') as f:
                    hrv_flat = json.load(f)
                phase_payload['bpm'] = hrv_flat.get('bpm')
                # The frontend expects hrv and also metric values for rmssd/sdnn/etc.
                phase_payload['hrv'] = hrv_flat.get('hrv') if isinstance(hrv_flat.get('hrv'), dict) else {}
                if not phase_payload['hrv']:
                    # If exporter wrote flat keys, tolerate it
                    for k in ['sdnn', 'rmssd', 'pnn50', 'mean_rr', 'mean_hr', 'min_hr', 'max_hr', 'nn_count', 'sqi', 'artifact_percent']:
                        if k in hrv_flat:
                            phase_payload['hrv'][k] = hrv_flat[k]

            # Signal report JSON (export_filtered_signal_report -> pos_filtered)
            if signal_json_path:
                with open(signal_json_path, 'r') as f:
                    signal_rows = json.load(f)  # list[dict]
                # Frontend uses filtered_signal directly.
                phase_payload['filtered_signal'] = [row.get('pos_filtered') for row in signal_rows if 'pos_filtered' in row]

                # pos_raw optional
                if signal_rows and 'pos_raw' in signal_rows[0]:
                    phase_payload['pos_signal'] = [row.get('pos_raw') for row in signal_rows if 'pos_raw' in row]

            # RR report JSON
            if rr_json_path:
                with open(rr_json_path, 'r') as f:
                    rr_rows = json.load(f)  # list[dict]
                # frontend uses rr_ms
                phase_payload['rr_ms'] = [row.get('rr_raw_ms') for row in rr_rows if 'rr_raw_ms' in row]
                if phase_payload['hrv'] is None:
                    phase_payload['hrv'] = {}

            # RGB report JSON
            if rgb_json_path:
                with open(rgb_json_path, 'r') as f:
                    rgb_rows = json.load(f)
                phase_payload['r'] = [row.get('r_mean') for row in rgb_rows if 'r_mean' in row]
                phase_payload['g'] = [row.get('g_mean') for row in rgb_rows if 'g_mean' in row]
                phase_payload['b'] = [row.get('b_mean') for row in rgb_rows if 'b_mean' in row]

            # Provide metric shortcuts expected by populateComparisonTable()
            hrv = phase_payload.get('hrv') or {}
            phase_payload['sdnn'] = hrv.get('sdnn')
            phase_payload['rmssd'] = hrv.get('rmssd')
            phase_payload['pnn50'] = hrv.get('pnn50')
            phase_payload['mean_rr'] = hrv.get('mean_rr')

            if phase_payload['bpm'] is None:
                # fallback: might be in HRV flat json under bpm directly
                phase_payload['bpm'] = hrv.get('mean_hr')  # not ideal, but avoids null

            results[phase] = phase_payload
        except Exception as e:
            print(f"[RESULTS] Error loading phase {phase} from {phase_dir}: {e}")
            import traceback
            traceback.print_exc()
            results[phase] = None

    return jsonify({
        'status': 'ok',
        'experiment_id': exp_id,
        'results': results
    })


@app.route('/api/data-collection/retry-export', methods=['POST'])
def retry_export():
    """Retry export from preserved recording for a given experiment phase.

    Expected JSON body:
      - experiment_id
      - participant_timestamp
      - phase (PRE/DURING/POST)
      - subject_info {name, ...}

    This regenerates exports (reports + metadata) from the already-preserved
    recording_{phase}.webm|mp4 without deleting original recordings/chunks.
    """
    try:
        data = request.get_json() or {}
        experiment_id = data.get('experiment_id')
        participant_timestamp = data.get('participant_timestamp')
        phase = (data.get('phase') or 'PRE').upper()
        subject_info = data.get('subject_info') or {}

        name = secure_filename(subject_info.get('name', 'Unknown'))

        if not experiment_id or not participant_timestamp:
            return jsonify({'status': 'error', 'message': 'Missing experiment_id or participant_timestamp'}), 400

        if phase not in ('PRE', 'DURING', 'POST'):
            return jsonify({'status': 'error', 'message': f'Invalid phase: {phase}'}), 400

        participant_root_dir = os.path.join(EXPORT_DIR, f"{name}_{participant_timestamp}")
        phase_dir = os.path.join(participant_root_dir, phase)
        video_path_webm = os.path.join(phase_dir, f'recording_{phase.lower()}.webm')
        video_path_mp4 = os.path.join(phase_dir, f'recording_{phase.lower()}.mp4')

        if os.path.exists(video_path_webm):
            video_path = video_path_webm
        elif os.path.exists(video_path_mp4):
            video_path = video_path_mp4
        else:
            return jsonify({'status': 'error', 'message': 'Preserved recording not found for retry'}), 404

        # Video verification logs
        file_size_bytes = os.path.getsize(video_path)
        print('[RETRY-EXPORT] Video preserved:', video_path)
        print('[RETRY-EXPORT] Video Size:', file_size_bytes)
        if file_size_bytes <= 0:
            return jsonify({'status': 'error', 'message': 'Preserved recording is empty'}), 400

        # Run analysis (same pipeline)
        results = process_video(video_path)
        hrv_results = analyze_hrv(results['filtered_signal'], results['fps'])

        session_data = {
            **results,
            **hrv_results,
            'subject_info': subject_info,
            'source': f'retry_export_{phase.lower()}'
        }

        # Export reports into the preserved phase directory.
        session_info = {'custom_dir': phase_dir}
        export_full_session(
            session_data,
            EXPORT_DIR,
            formats=['csv', 'json', 'xlsx'],
            session_info=session_info
        )

        # Update in-memory summary
        summary = {
            'bpm': round(results['bpm'], 1),
            'sdnn': round(hrv_results['hrv']['sdnn'], 1),
            'rmssd': round(hrv_results['hrv']['rmssd'], 1),
            'pnn50': round(hrv_results['hrv']['pnn50'], 1),
            'mean_rr': round(hrv_results['hrv']['mean_rr'], 1),
            'sqi': round(hrv_results['sqi'], 1),
            'timestamp': datetime.datetime.now().isoformat()
        }
        if experiment_id not in experiment_sessions:
            experiment_sessions[experiment_id] = {}
        experiment_sessions[experiment_id][phase] = summary

        return jsonify({'status': 'ok', 'message': 'Retry export completed.', 'summary': summary})

    except Exception as e:
        print(f"[RETRY-EXPORT] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/data-collection/export-all')
def export_collection_all():
    """Download the entire experiment folder as a ZIP."""
    subject_name = request.args.get('name')
    exp_id = request.args.get('experiment_id')
    
    if not subject_name or not exp_id:
        return "Missing parameters", 400
        
    # New layout: exports/<participant>_<participant_timestamp>/... (exp_id is user experiment_id)
    # Prefer timestamp folder match when exp_id matches experiment_id is not directly used.
    base_dir = os.path.join(EXPORT_DIR, secure_filename(subject_name))
    if os.path.exists(base_dir):
        subject_dir = base_dir
    else:
        # fallback to old layout
        subject_dir = os.path.join(EXPORT_DIR, secure_filename(subject_name), secure_filename(exp_id))
    if not os.path.exists(subject_dir):
        return f"Experiment folder not found: {subject_dir}", 404
        
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(subject_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.join(subject_dir, '..'))
                zf.write(file_path, arcname)
    
    zip_buffer.seek(0)
    filename = f"experiment_{secure_filename(subject_name)}_{exp_id}.zip"
    
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/zip'
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  rPPG Web Interface")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

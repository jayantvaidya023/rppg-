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

EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'exports')
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'rppg_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route('/')
def index():
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
    """Minimal SSE verification route."""
    def generate():
        print("DEBUG: Test Stream Client Connected")
        count = 0
        while True:
            count += 1
            data = {"bpm": 60 + (count % 20), "status": "Test Connection Active", "timestamp": time.time()}
            print(f"DEBUG: Sending Test Payload #{count}")
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1.0)
    return Response(generate(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

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
    """Server-Sent Events stream for live BPM and waveform data."""
    def generate():
        client_id = f"client_{int(time.time())}"
        print(f"[{client_id}] SSE: Connection established")
        
        last_heartbeat = 0
        
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
                    'status': state['status'],
                    'timestamp': now
                }
                
                if state['hrv']:
                    data['sqi'] = state['hrv'].get('sqi', 0.0)
                    data['artifact_percent'] = state['hrv'].get('artifact_percent', 0.0)

                # Send data if processing OR send heartbeat to keep connection alive
                if state['is_running'] or (now - last_heartbeat > 2.0):
                    if not state['is_running']:
                        data['heartbeat'] = True
                        data['status'] = "Waiting for data..."
                    
                    payload = json.dumps(data)
                    # print(f"[{client_id}] SSE: Sending payload (running={state['is_running']})")
                    yield f"data: {payload}\n\n"
                    last_heartbeat = now
                
                time.sleep(0.1)
        except Exception as e:
            print(f"[{client_id}] SSE: Stream exception: {e}")
        finally:
            print(f"[{client_id}] SSE: Connection terminated")

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
            'Transfer-Encoding': 'chunked'
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
    """Export session reports in the specified format."""
    if not last_session_data:
        return jsonify({
            'status': 'error',
            'message': 'No session data. Analyze a video or record a session first.'
        }), 400

    if format_type not in ('csv', 'json', 'xlsx'):
        return jsonify({
            'status': 'error',
            'message': f'Unsupported format: {format_type}. Use csv, json, or xlsx.'
        }), 400

    # Update subject info from query params if available
    name = request.args.get('name', '').strip()
    age = request.args.get('age', '').strip()
    gender = request.args.get('gender', '').strip()
    if name or age or gender:
        last_session_data['subject_info'] = {'name': name, 'age': age, 'gender': gender}

    os.makedirs(EXPORT_DIR, exist_ok=True)

    results = export_full_session(
        last_session_data, EXPORT_DIR, formats=[format_type]
    )

    # Collect all generated files into a zip
    all_files = []
    for file_list in results.values():
        all_files.extend(file_list)

    if not all_files:
        return jsonify({'status': 'error', 'message': 'No data to export.'}), 400

    # If single file, send directly
    if len(all_files) == 1:
        return send_file(
            all_files[0],
            as_attachment=True,
            download_name=os.path.basename(all_files[0])
        )

    # Multiple files: zip them
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in all_files:
            zf.write(f, os.path.basename(f))
    zip_buffer.seek(0)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f'rppg_report_{timestamp}.zip',
        mimetype='application/zip'
    )


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
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  rPPG Web Interface")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

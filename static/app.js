/* app.js — rPPG Monitor Dashboard Logic */

// State
let eventSource = null;
let waveformData = [];
let hrTimeData = { times: [], values: [] };

// Client-side recording state
let localStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordStartTime = 0;
let timerInterval = null;

// Quality tracking state
let qualityCheckInterval = null;
let framesCount = 0;
let lastFpsTime = 0;
let qualityCanvas = document.createElement('canvas');
let qualityCtx = qualityCanvas.getContext('2d', { willReadFrequently: true });
qualityCanvas.width = 160;
qualityCanvas.height = 120;

// DOM refs
const bpmValue = document.getElementById('bpmValue');
const bpmRange = document.getElementById('bpmRange');
const bpmMethod = document.getElementById('bpmMethod');
const bpmCard = document.getElementById('bpmCard');
const statusBadge = document.getElementById('statusBadge');
const statusMessage = document.getElementById('statusMessage');
const sourceSelect = document.getElementById('sourceSelect');
const ipInput = document.getElementById('ipInput');
const waveformCanvas = document.getElementById('waveformCanvas');
const hrTimeCanvas = document.getElementById('hrTimeCanvas');

// HRV refs
const sdnnValue = document.getElementById('sdnnValue');
const rmssdValue = document.getElementById('rmssdValue');
const pnn50Value = document.getElementById('pnn50Value');
const meanRRValue = document.getElementById('meanRRValue');
const peakBpmValue = document.getElementById('peakBpmValue');
const nnCountValue = document.getElementById('nnCountValue');

// Source selector toggle
sourceSelect.addEventListener('change', function() {
    const val = this.value;
    document.getElementById('videoControls').style.display = val === 'file' ? 'block' : 'none';
    document.getElementById('defaultControls').style.display = val === 'default' ? 'block' : 'none';
    document.getElementById('cameraControls').style.display = (val === 'file' || val === 'default') ? 'none' : 'block';
    document.getElementById('recordControls').style.display = (val === 'file' || val === 'default') ? 'none' : 'block';
    
    // Stop local camera if switching to file
    if ((val === 'file' || val === 'default') && localStream) {
        stopCamera();
    }
});

// ---- Quality Checks (Light & FPS) ----
function checkQuality() {
    if (!localStream) return;
    
    const video = document.getElementById('localVideo');
    if (video.videoWidth === 0) return; // not ready

    // FPS Calculation
    const now = performance.now();
    framesCount++;
    if (now - lastFpsTime >= 1000) {
        document.getElementById('fpsBadge').textContent = `FPS: ${framesCount}`;
        framesCount = 0;
        lastFpsTime = now;
    }

    // Light Calculation
    qualityCtx.drawImage(video, 0, 0, qualityCanvas.width, qualityCanvas.height);
    const imageData = qualityCtx.getImageData(0, 0, qualityCanvas.width, qualityCanvas.height);
    const data = imageData.data;
    let brightnessSum = 0;
    
    // sample pixels
    const sampleStep = 4 * 4; // every 4th pixel
    let count = 0;
    for (let i = 0; i < data.length; i += sampleStep) {
        const r = data[i], g = data[i+1], b = data[i+2];
        brightnessSum += (0.299 * r + 0.587 * g + 0.114 * b);
        count++;
    }
    
    const avgBrightness = brightnessSum / count;
    document.getElementById('lightBadge').textContent = `Light: ${Math.round(avgBrightness)}`;
    
    const warningText = document.getElementById('qualityWarning');
    if (avgBrightness < 50) {
        document.getElementById('lightBadge').style.color = '#f87171'; // red
        warningText.style.display = 'block';
        warningText.textContent = "Too dark! Please move to a brighter environment.";
    } else if (avgBrightness > 220) {
        document.getElementById('lightBadge').style.color = '#fbbf24'; // amber
        warningText.style.display = 'block';
        warningText.textContent = "Too bright/washed out. Reduce backlight.";
    } else {
        document.getElementById('lightBadge').style.color = '#34d399'; // green
        warningText.style.display = 'none';
    }
    
    qualityCheckInterval = requestAnimationFrame(checkQuality);
}

// ---- Camera Controls ----
async function startCamera() {
    const sourceVal = sourceSelect.value;
    if (sourceVal === 'file' || sourceVal === 'default') return;

    try {
        const constraints = {
            video: { 
                facingMode: sourceVal === 'front' ? 'user' : 'environment',
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            },
            audio: false
        };

        localStream = await navigator.mediaDevices.getUserMedia(constraints);
        const video = document.getElementById('localVideo');
        video.srcObject = localStream;
        
        // Hide legacy image stream, show local video
        document.getElementById('videoStream').style.display = 'none';
        video.style.display = 'block';
        
        // Flip preview if front camera
        video.style.transform = sourceVal === 'front' ? 'scaleX(-1)' : 'none';
        
        document.getElementById('cameraPreviewCard').style.display = 'block';
        document.getElementById('btnStartCamera').disabled = true;
        document.getElementById('btnStopCamera').disabled = false;
        document.getElementById('btnStartRecord').disabled = false;
        
        statusBadge.textContent = 'Camera Ready';
        statusBadge.className = 'status-badge active';
        setStatus('Camera started. Position your face in the oval and ensure good lighting.');
        
        lastFpsTime = performance.now();
        framesCount = 0;
        qualityCheckInterval = requestAnimationFrame(checkQuality);

    } catch (err) {
        console.error(err);
        setStatus('Camera Error: ' + err.message + '. Please allow camera permissions.');
    }
}

function stopCamera() {
    if (localStream) {
        localStream.getTracks().forEach(track => track.stop());
        localStream = null;
    }
    if (qualityCheckInterval) cancelAnimationFrame(qualityCheckInterval);
    
    document.getElementById('localVideo').srcObject = null;
    document.getElementById('cameraPreviewCard').style.display = 'none';
    
    document.getElementById('btnStartCamera').disabled = false;
    document.getElementById('btnStopCamera').disabled = true;
    document.getElementById('btnStartRecord').disabled = true;
    document.getElementById('btnStopRecord').disabled = true;
    
    statusBadge.textContent = 'Idle';
    statusBadge.className = 'status-badge';
    setStatus('Camera stopped.');
}

// ---- Subject Info ----
function getSubjectInfo() {
    return {
        name: document.getElementById('subjectName').value.trim(),
        age: document.getElementById('subjectAge').value,
        gender: document.getElementById('subjectGender').value
    };
}

// ---- Recording ----
function startRecording() {
    if (!localStream) {
        setStatus("Start camera first!");
        return;
    }

    recordedChunks = [];
    
    // Try to use a format that OpenCV can easily process on the backend
    let options = { mimeType: 'video/webm;codecs=vp8' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options = { mimeType: 'video/webm' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options = { mimeType: 'video/mp4' }; // Safari fallback
        }
    }

    try {
        mediaRecorder = new MediaRecorder(localStream, options);
    } catch (e) {
        console.error('MediaRecorder error:', e);
        setStatus("Error creating MediaRecorder. " + e.message);
        return;
    }

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordedChunks.push(event.data);
    };

    mediaRecorder.onstop = uploadRecording;

    mediaRecorder.start();
    
    // Timer
    recordStartTime = Date.now();
    const timerDiv = document.getElementById('recordingTimer');
    timerDiv.style.display = 'block';
    
    timerInterval = setInterval(() => {
        const seconds = Math.floor((Date.now() - recordStartTime) / 1000);
        const mm = String(Math.floor(seconds / 60)).padStart(2, '0');
        const ss = String(seconds % 60).padStart(2, '0');
        timerDiv.textContent = `${mm}:${ss}`;
    }, 1000);

    setStatus('Recording session locally...');
    document.getElementById('btnStartRecord').disabled = true;
    document.getElementById('btnStopRecord').disabled = false;
    document.getElementById('btnStopCamera').disabled = true; // prevent stopping cam while recording
    statusBadge.textContent = 'Recording';
    statusBadge.className = 'status-badge recording';
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        clearInterval(timerInterval);
        document.getElementById('recordingTimer').style.display = 'none';
        
        document.getElementById('btnStopRecord').disabled = true;
        document.getElementById('btnStopCamera').disabled = false;
        setStatus('Finalizing recording...');
    }
}

function uploadRecording() {
    // Create Blob from chunks
    let mimeType = mediaRecorder.mimeType || 'video/webm';
    const blob = new Blob(recordedChunks, { type: mimeType });
    const ext = mimeType.includes('mp4') ? 'mp4' : 'webm';
    
    // Prepare form data
    const formData = new FormData();
    formData.append('video', blob, `recording.${ext}`);
    formData.append('subject_info', JSON.stringify(getSubjectInfo()));

    // Show upload progress UI
    document.getElementById('uploadProgressContainer').style.display = 'block';
    document.getElementById('uploadProgressBar').style.width = '20%'; // fake initial progress
    setStatus('Uploading video for analysis...');

    // Use /api/analyze-video endpoint
    fetch('/api/analyze-video', {
        method: 'POST',
        body: formData
    })
    .then(r => r.json())
    .then(data => {
        document.getElementById('uploadProgressBar').style.width = '100%';
        if (data.status === 'ok') {
            handleAnalysisResult(data);
        } else {
            setStatus('Server Error: ' + data.message);
            document.getElementById('uploadProgressContainer').style.display = 'none';
        }
    })
    .catch(err => {
        console.error(err);
        setStatus('Upload failed: ' + err.message);
        document.getElementById('uploadProgressContainer').style.display = 'none';
    });
}

// ---- Video Analysis ----
function analyzeVideo() {
    const fileInput = document.getElementById('videoFile');
    if (!fileInput.files.length) { setStatus('Select a video file.'); return; }

    const formData = new FormData();
    formData.append('video', fileInput.files[0]);
    formData.append('subject_info', JSON.stringify(getSubjectInfo()));

    setStatus('Analyzing video...');
    fetch('/api/analyze-video', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(handleAnalysisResult)
    .catch(err => setStatus('Error: ' + err));
}

function analyzeDefault() {
    setStatus('Analyzing rPPG_video.mp4...');
    fetch('/api/analyze-default', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject_info: getSubjectInfo() })
    })
    .then(r => r.json())
    .then(handleAnalysisResult)
    .catch(err => setStatus('Error: ' + err));
}

function handleAnalysisResult(data) {
    if (data.status === 'ok') {
        setStatus(data.message);
        // Switch view to stream coming from server (for processing feedback)
        document.getElementById('localVideo').style.display = 'none';
        document.getElementById('videoStream').style.display = 'block';
        document.getElementById('videoStream').src = '/api/video_feed?t=' + new Date().getTime();
        document.getElementById('cameraPreviewCard').style.display = 'block';
        
        // Cleanup UI
        document.getElementById('uploadProgressContainer').style.display = 'none';
        document.getElementById('uploadProgressBar').style.width = '0%';
        
        statusBadge.textContent = 'Processing...';
        statusBadge.className = 'status-badge active';
        startSSE();
    } else {
        setStatus('Error: ' + data.message);
        document.getElementById('uploadProgressContainer').style.display = 'none';
    }
}

// ---- SSE Stream ----
function startSSE() {
    if (eventSource) eventSource.close();
    eventSource = new EventSource('/api/stream');
    let startTime = Date.now();
    hrTimeData = { times: [], values: [] };
    
    eventSource.onmessage = function(e) {
        try {
            const data = JSON.parse(e.data);
            
            if (data.status) {
                setStatus(data.status);
                if (data.status === "Analysis complete") {
                    stopSSE();
                    statusBadge.textContent = 'Analysis Complete';
                    statusBadge.className = 'status-badge';
                    document.getElementById('btnStartCamera').disabled = false;
                    document.getElementById('btnStopCamera').disabled = true;
                    document.getElementById('btnStartRecord').disabled = true;
                    document.getElementById('btnStopRecord').disabled = true;
                    
                    // fetch final summary to update UI
                    fetch('/api/session-summary').then(r=>r.json()).then(summary => {
                        if (summary.status === 'ok') {
                            setStatus(`Analysis Complete: ${summary.bpm} BPM | ${summary.frames} frames`);
                            updateBPM(summary.bpm);
                            updateHRV(summary.hrv, summary.peak_bpm);
                        }
                    });
                }
            }

            if (data.bpm > 0 && data.status !== "Analysis complete") {
                updateBPM(data.bpm);
                let elapsedSec = (Date.now() - startTime) / 1000;
                hrTimeData.times.push(elapsedSec);
                hrTimeData.values.push(data.bpm);
                if (hrTimeData.times.length > 100) {
                    hrTimeData.times.shift();
                    hrTimeData.values.shift();
                }
                drawHRTime(hrTimeCanvas, hrTimeData.times, hrTimeData.values);
            }
            if (data.hrv && data.hrv.mean_rr > 0) {
                updateHRV(data.hrv, 0);
            }
            if (data.sqi !== undefined) {
                document.getElementById('sqiValue').textContent = `SQI: ${data.sqi.toFixed(1)}%`;
                if (data.sqi > 80) document.getElementById('sqiValue').style.color = '#4cd137';
                else if (data.sqi > 50) document.getElementById('sqiValue').style.color = '#fbc531';
                else document.getElementById('sqiValue').style.color = '#e84118';
            }
            if (data.artifact_percent !== undefined) {
                document.getElementById('artifactValue').textContent = `Artifacts: ${data.artifact_percent.toFixed(1)}%`;
                if (data.artifact_percent < 5) document.getElementById('artifactValue').style.color = '#4cd137';
                else if (data.artifact_percent < 15) document.getElementById('artifactValue').style.color = '#fbc531';
                else document.getElementById('artifactValue').style.color = '#e84118';
            }
            if (data.waveform && data.waveform.length > 0) {
                waveformData = data.waveform;
                drawWaveform(waveformCanvas, waveformData, '#34d399');
            }
        } catch(err) {}
    };
}

function stopSSE() {
    if (eventSource) { eventSource.close(); eventSource = null; }
}

// ---- UI Updates ----
function updateBPM(bpm) {
    bpmValue.textContent = Math.round(bpm);
    bpmValue.className = 'bpm-value';
    if (bpm > 100) bpmValue.classList.add('elevated');
    if (bpm > 120) { bpmValue.classList.remove('elevated'); bpmValue.classList.add('high'); }
}

function updateHRV(hrv, peakBpm) {
    if (!hrv) return;
    sdnnValue.textContent = hrv.sdnn ? hrv.sdnn.toFixed(1) : '--';
    rmssdValue.textContent = hrv.rmssd ? hrv.rmssd.toFixed(1) : '--';
    pnn50Value.textContent = hrv.pnn50 ? hrv.pnn50.toFixed(1) : '--';
    meanRRValue.textContent = hrv.mean_rr ? hrv.mean_rr.toFixed(0) : '--';
    peakBpmValue.textContent = peakBpm ? peakBpm.toFixed(1) : (hrv.mean_hr ? hrv.mean_hr.toFixed(1) : '--');
    nnCountValue.textContent = hrv.nn_count || '--';
    if (hrv.min_hr && hrv.max_hr) {
        bpmRange.textContent = `Min ${hrv.min_hr.toFixed(0)} | Max ${hrv.max_hr.toFixed(0)}`;
    }
}

function setStatus(msg) {
    statusMessage.textContent = msg;
}

// ---- Canvas Drawing ----
function drawWaveform(canvas, data, color) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);

    if (data.length < 2) return;

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const pad = 10;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';

    for (let i = 0; i < data.length; i++) {
        const x = (i / (data.length - 1)) * (w - 2 * pad) + pad;
        const y = h - pad - ((data[i] - min) / range) * (h - 2 * pad);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Glow effect
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;
}

function drawHRTime(canvas, times, values) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);

    if (values.length < 2) return;

    const pad = 10;
    const minVal = 40;
    const maxVal = 180;
    const range = maxVal - minVal;

    // Grid lines
    ctx.strokeStyle = 'rgba(80,120,200,0.1)';
    ctx.lineWidth = 0.5;
    for (let bpm = 60; bpm <= 160; bpm += 20) {
        const y = h - pad - ((bpm - minVal) / range) * (h - 2 * pad);
        ctx.beginPath();
        ctx.moveTo(pad, y);
        ctx.lineTo(w - pad, y);
        ctx.stroke();
        ctx.fillStyle = 'rgba(80,120,200,0.3)';
        ctx.font = '10px Inter';
        ctx.fillText(bpm + '', 2, y - 2);
    }

    // Line
    ctx.beginPath();
    ctx.strokeStyle = '#f87171';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';

    const tMin = times[0];
    const tMax = times[times.length - 1];
    const tRange = tMax - tMin || 1;

    for (let i = 0; i < values.length; i++) {
        const x = ((times[i] - tMin) / tRange) * (w - 2 * pad) + pad;
        const y = h - pad - ((values[i] - minVal) / range) * (h - 2 * pad);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowColor = '#f87171';
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;
}

// ---- Export ----
function exportReport(format) {
    const info = getSubjectInfo();
    const params = new URLSearchParams(info).toString();
    setStatus(`Exporting ${format.toUpperCase()} report...`);
    window.location.href = `/api/export/${format}?${params}`;
    setTimeout(() => setStatus('Export download started.'), 1000);
}

// Init: trigger source select
sourceSelect.dispatchEvent(new Event('change'));

/**
 * data_collection.js
 * Manages the multi-phase physiological data collection workflow.
 */

let mediaRecorder;
let recordedChunks = [];
let currentPhase = 'PRE'; // PRE, DURING, POST (DEFAULT: ALWAYS PRE)
let timerInterval;
let startTime;
const PHASE_DURATION = 300; // 5 minutes in seconds

let experimentMetadata = {
    id: '',
    participantTimestamp: '',
    subject: {
        name: '',
        age: '',
        gender: '',
        notes: ''
    }
};


let phaseResults = {
    PRE: null,
    DURING: null,
    POST: null
};

// --- New Robust Pipeline State ---
let chunkIndex = 0;
let uploadQueue = [];
let isUploading = false;
let db;
let currentTaskId = null;
let pollingInterval = null;

// Ensures POST finalization can't be missed if MediaRecorder.onstop doesn't fire reliably
let hasFinalizedCurrentPhase = false;

// Unified Dashboard State
let fullExperimentResults = null;
let charts = {
    bpmComp: null,
    hrvComp: null,
    waveform: null,
    rr: null
};

// Initialize IndexedDB for offline chunk storage
const dbRequest = indexedDB.open("rPPG_Collection", 1);
dbRequest.onupgradeneeded = (e) => {
    db = e.target.result;
    if (!db.objectStoreNames.contains("chunks")) {
        db.createObjectStore("chunks", { keyPath: "id" });
    }
};
dbRequest.onsuccess = (e) => { db = e.target.result; };

// --- Initialization ---

async function startExperiment() {
    // New experiment always starts cleanly from PRE (no resume).
    resetWorkflow(false);

    const name = document.getElementById('subjectName').value.trim();
    const expId = document.getElementById('experimentId').value.trim();
    
    if (!name || !expId) {
        alert("Please provide Subject Name and Experiment ID.");
        return;
    }

    experimentMetadata.id = expId;
    // Create a consistent participant package timestamp for PRE/DURING/POST.
    experimentMetadata.participantTimestamp = new Date().toISOString().replace(/[:.]/g, '-');
    experimentMetadata.subject = {
        name: name,
        age: document.getElementById('subjectAge').value,
        gender: document.getElementById('subjectGender').value,
        notes: document.getElementById('subjectNotes').value
    };

    // Transition to PRE phase
    switchPhase('record');
    updateStatus(`Experiment ${expId} initialized. Ready for PRE session.`);
    
    // Request Camera Access
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'user',
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 }
            }, 
            audio: false 
        });
        document.getElementById('recordVideo').srcObject = stream;
    } catch (err) {
        console.error("Camera access error:", err);
        alert("Camera access is required for this experiment.");
    }

    // Save state for recovery
    saveAppState();
}

function saveAppState() {
    localStorage.setItem('rPPG_Experiment', JSON.stringify({
        metadata: experimentMetadata,
        currentPhase: currentPhase,
        phaseResults: phaseResults,
        timestamp: Date.now()
    }));
}

function hasIncompleteExperimentSaved(data) {
    if (!data || !data.metadata || !data.phaseResults) return false;

    // Only consider as incomplete if at least one phase result is missing.
    const pr = data.phaseResults || {};
    const hasAnyPhaseResult = Boolean(pr.PRE || pr.DURING || pr.POST);
    const allDone = Boolean(pr.PRE && pr.DURING && pr.POST);
    return hasAnyPhaseResult && !allDone;
}

function loadAppState() {
    const saved = localStorage.getItem('rPPG_Experiment');
    if (!saved) return;

    let data;
    try {
        data = JSON.parse(saved);
    } catch {
        return;
    }

    // Never auto-resume. We only preload metadata fields when the user explicitly resumes.
    // Still validate that the save is recent.
    if (!data.timestamp || Date.now() - data.timestamp >= 2 * 60 * 60 * 1000) return;

    if (hasIncompleteExperimentSaved(data)) {
        // Stash for explicit resume.
        window.__rPPG_SAVED_STATE__ = data;
        document.getElementById('statusMessage').innerText = 'Recovered an incomplete experiment. Click "Resume Experiment" to continue.';
    }

    // Always start cleanly from PRE.
    resetWorkflow(true);
}

function resetWorkflow(fromLoad = false) {

    // Clear all runtime workflow state
    currentPhase = 'PRE';
    phaseResults = { PRE: null, DURING: null, POST: null };
    experimentMetadata = {
        id: '',
        subject: { name: '', age: '', gender: '', notes: '' }
    };

    fullExperimentResults = null;
    charts = { bpmComp: null, hrvComp: null, waveform: null, rr: null };

    chunkIndex = 0;
    uploadQueue = [];
    isUploading = false;
    currentTaskId = null;

    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }

    if (!fromLoad) {
        // Clear persisted workflow state (but keep IndexedDB chunks as a cache; backend will handle missing chunks)
        localStorage.removeItem('rPPG_Experiment');
        window.__rPPG_SAVED_STATE__ = null;
    }

    hasFinalizedCurrentPhase = false;

    // Reset UI: go back to Subject/recording PRE step.
    // Existing markup uses ids: phase-info, phase-record, phase-summary; and steps: step-info, step-pre etc.
    try {
        switchPhase('record');
        document.getElementById('phase-info')?.classList.add('active');
        document.getElementById('phase-record')?.classList.remove('active');
        document.getElementById('phase-summary')?.classList.remove('active');

        // Stepper: ensure SUBECT active; step ids are step-info, step-pre, step-during, step-post, step-summary.
        document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
        document.getElementById('step-info')?.classList.add('active');

        // Hide processing overlay if present
        document.getElementById('processingOverlay')?.classList.remove('active');
    } catch {
        // ignore UI reset failures
    }
}


window.addEventListener('DOMContentLoaded', () => {
    // Always ensure we start at PRE, never auto-resume.
    // loadAppState() will stash an incomplete state (if any), and UI will offer explicit resume.
    loadAppState();
    window.__rPPG_UI__maybeEnableResumeButton?.();
});


function switchPhase(phaseId) {
    document.querySelectorAll('.phase-container').forEach(p => p.classList.remove('active'));
    document.getElementById(`phase-${phaseId}`).classList.add('active');
    
    // Update Stepper
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    if (phaseId === 'record') {
        document.getElementById(`step-${currentPhase.toLowerCase()}`).classList.add('active');
    } else {
        document.getElementById(`step-${phaseId}`).classList.add('active');
    }
}

// --- Recording & Timer ---

function handlePhaseAction() {
    const btn = document.getElementById('btnAction');
    
    if (btn.innerText.includes('Start')) {
        startRecording();
    } else if (btn.innerText.includes('Stop')) {
        stopRecording();
    } else if (btn.innerText.includes('Proceed') || btn.innerText.includes('View')) {
        prepareNextPhase();
    }
}

function startRecording() {
    recordedChunks = [];
    hasFinalizedCurrentPhase = false;

    const stream = document.getElementById('recordVideo').srcObject;
    
    let mimeType = 'video/webm;codecs=vp8';
    if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'video/mp4;codecs=avc1';
    }
    if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = '';
    }
    
    console.log(`[REC] Using MIME type: ${mimeType || 'default'}`);
    mediaRecorder = new MediaRecorder(stream, { mimeType });
    
    chunkIndex = 0;
    mediaRecorder.ondataavailable = async (e) => {
        if (e.data.size > 0) {
            const chunk = {
                id: `${experimentMetadata.id}_${currentPhase}_${chunkIndex}`,
                blob: e.data,
                index: chunkIndex,
                phase: currentPhase,
                expId: experimentMetadata.id
            };
            
            saveChunkToDB(chunk);
            uploadQueue.push(chunk);
            processUploadQueue();
            chunkIndex++;
        }
    };

    mediaRecorder.onstop = finalizeSession;
    
    mediaRecorder.start(3000); // Shorter chunks for more frequent syncing
    
    document.getElementById('recordingIndicator').style.display = 'block';
    document.getElementById('roiGuide').style.display = 'flex';
    document.getElementById('syncStatus').style.display = 'flex';
    
    const btn = document.getElementById('btnAction');
    btn.innerText = `Stop Recording (${currentPhase})`;
    btn.classList.remove('btn-record', 'btn-start');
    btn.classList.add('btn-stop');
    
    startTimer(PHASE_DURATION);
    updateStatus(`Recording ${currentPhase} session...`);
}

function stopRecording() {
    console.log("[REC] Stopping recording for phase:", currentPhase);

    // Always stop timer first to prevent re-entrancy
    clearInterval(timerInterval);

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        console.log("[REC] MediaRecorder stopped");
    }

    document.getElementById('recordingIndicator').style.display = 'none';
    document.getElementById('roiGuide').style.display = 'none';

    // POST-specific reliability: some browsers occasionally don't fire onstop.
    // Ensure finalizeSession still runs so UI doesn't get stuck on "Recording POST session...".
    if (currentPhase === 'POST') {
        setTimeout(() => {
            if (hasFinalizedCurrentPhase) return;

            // If recorder hasn't transitioned to inactive yet, we still proceed to avoid deadlock.
            console.warn("[POST] onstop reliability fallback: forcing finalizeSession()");
            finalizeSession();
        }, 1200);
    }
}

function startTimer(seconds) {
    let timeLeft = seconds;
    const display = document.getElementById('timerDisplay');
    const bar = document.getElementById('progressBarFill');
    
    display.innerText = formatTime(timeLeft);
    bar.style.width = '0%';
    
    timerInterval = setInterval(() => {
        timeLeft--;
        display.innerText = formatTime(timeLeft);
        
        const progress = ((seconds - timeLeft) / seconds) * 100;
        bar.style.width = `${progress}%`;
        
        if (timeLeft <= 0) {
            stopRecording();
        }
    }, 1000);
}

// --- Chunk & Upload Management ---

async function saveChunkToDB(chunk) {
    if (!db) return;
    const tx = db.transaction("chunks", "readwrite");
    tx.objectStore("chunks").put(chunk);
}

async function processUploadQueue() {
    if (isUploading || uploadQueue.length === 0) return;
    
    isUploading = true;
    const chunk = uploadQueue[0];
    
    const formData = new FormData();
    formData.append('video', chunk.blob, `chunk_${chunk.index}.blob`);
    formData.append('experiment_id', chunk.expId);
    formData.append('phase', chunk.phase);
    formData.append('chunk_index', chunk.index);
    
    try {
        console.log(`[UPLOAD] Uploading chunk ${chunk.index}`);
        const response = await fetch('/api/data-collection/upload-chunk', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            console.log(`[UPLOAD] Chunk ${chunk.index} uploaded successfully`);
            uploadQueue.shift();
        } else {
            console.error(`[UPLOAD] Upload failed for chunk ${chunk.index}:`, response.status, response.statusText);
            // Retry after delay
            setTimeout(() => {
                isUploading = false;
                processUploadQueue();
            }, 2000);
            return;
        }
    } catch (err) {
        console.error(`[UPLOAD] Connection error for chunk ${chunk.index}:`, err);
        // Retry after delay
        setTimeout(() => {
            isUploading = false;
            processUploadQueue();
        }, 2000);
        return;
    }
    
    isUploading = false;
    if (uploadQueue.length > 0) {
        setTimeout(processUploadQueue, 500);
    } else {
        console.log("[UPLOAD] All chunks uploaded");
        document.getElementById('syncStatus').style.display = 'none';
    }
}

async function finalizeSession() {
    if (hasFinalizedCurrentPhase) {
        console.warn("[FINALIZE] finalizeSession already ran for phase:", currentPhase);
        return;
    }
    hasFinalizedCurrentPhase = true;

    console.log("[FINALIZE] Recording stopped, finalizing session for phase:", currentPhase);
    
    // Show Processing Overlay
    const overlay = document.getElementById('processingOverlay');
    const title = document.getElementById('processingTitle');
    const subtitle = document.getElementById('processingSubtitle');
    const bar = document.getElementById('processingBar');
    
    if (!overlay) {
        console.error("[FINALIZE] processingOverlay element not found!");
        alert("Error: Processing overlay UI missing");
        return;
    }
    
    overlay.classList.add('active');
    title.innerText = `Finalizing ${currentPhase} Session`;
    subtitle.innerText = "Syncing remaining data chunks...";
    bar.style.width = '20%';

    // Wait for upload queue to clear
    let timeout = 30; // 30 seconds max wait
    while (uploadQueue.length > 0 && timeout > 0) {
        subtitle.innerText = `Syncing: ${uploadQueue.length} chunks remaining...`;
        console.log(`[FINALIZE] Waiting for uploads: ${uploadQueue.length} chunks remaining`);
        await new Promise(r => setTimeout(r, 1000));
        timeout--;
        bar.style.width = `${20 + (30-timeout)*2}%`;
    }
    
    if (uploadQueue.length > 0) {
        console.error("[FINALIZE] Upload queue did not clear, proceeding anyway");
    } else {
        console.log("[FINALIZE] Upload queue cleared");
    }
    
    bar.style.width = '50%';
    subtitle.innerText = "Requesting physiological analysis...";

    try {
        console.log("[FINALIZE] Sending finalize request for phase:", currentPhase);
    const response = await fetch('/api/data-collection/finalize-session', {
        method: 'POST',

            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                experiment_id: experimentMetadata.id,
                participant_timestamp: experimentMetadata.participantTimestamp,
                phase: currentPhase,
                subject_info: experimentMetadata.subject
            })
        });
        
        const result = await response.json();
        console.log("[FINALIZE] Finalize response:", result);
        if (result.status === 'ok') {
            currentTaskId = result.task_id;
            bar.style.width = '60%';
            startPollingStatus();
        } else {
            console.error("[FINALIZE] Finalization failed:", result.message);
            alert(`Finalization failed: ${result.message}`);
            hideProcessingOverlay();
        }
    } catch (err) {
        console.error("[FINALIZE] Finalize error:", err);
        alert("Failed to finalize session. Check connection.");
        hideProcessingOverlay();
    }
}

function startPollingStatus() {
    const subtitle = document.getElementById('processingSubtitle');
    const bar = document.getElementById('processingBar');
    
    if (pollingInterval) clearInterval(pollingInterval);
    
    console.log(`[POLL] Starting polling for task ${currentTaskId}`);
    let pollCount = 0;
    const maxPolls = 180; // 6 minutes max (180 * 2 second intervals)
    
    pollingInterval = setInterval(async () => {
        pollCount++;
        try {
            const response = await fetch(`/api/data-collection/task-status/${currentTaskId}`);
            const status = await response.json();
            console.log(`[POLL] Task status (poll #${pollCount}):`, status);
            
            if (status.status === 'completed') {
                clearInterval(pollingInterval);
                console.log(`[POLL] Task completed on poll #${pollCount}`);
                phaseResults[currentPhase] = status.result;
                saveAppState();
                bar.style.width = '100%';
                subtitle.innerText = "Analysis complete!";
                
                setTimeout(() => {
                    hideProcessingOverlay();
                    console.log(`[POLL] Calling autoTransition from polling completion`);
                    autoTransition();
                }, 1000);
            } else if (status.status === 'failed') {
                clearInterval(pollingInterval);
                console.error(`[POLL] Task failed:`, status.error);
                alert(`Analysis failed: ${status.error}`);
                hideProcessingOverlay();
            } else if (pollCount >= maxPolls) {
                clearInterval(pollingInterval);
                console.error(`[POLL] Task timeout after ${maxPolls} polls (${maxPolls * 2} seconds)`);
                alert(`Processing timeout. Click OK to continue.`);
                hideProcessingOverlay();
                // Force progress forward anyway
                setTimeout(() => {
                    autoTransition();
                }, 500);
            } else {
                subtitle.innerText = `Processing ${currentPhase} signals... (${pollCount} attempts)`;
                let currentWidth = parseFloat(bar.style.width);
                if (currentWidth < 95) bar.style.width = `${currentWidth + 0.5}%`;
            }
        } catch (err) {
            console.error("[POLL] Status polling error:", err);
            if (pollCount >= maxPolls) {
                clearInterval(pollingInterval);
                alert(`Network error. Click OK to continue.`);
                hideProcessingOverlay();
                setTimeout(() => {
                    autoTransition();
                }, 500);
            }
        }
    }, 2000);
}

function hideProcessingOverlay() {
    document.getElementById('processingOverlay').classList.remove('active');
}

function autoTransition() {
    console.log(`[TRANSITION] autoTransition called, currentPhase=${currentPhase}`);
    document.getElementById(`step-${currentPhase.toLowerCase()}`).classList.add('completed');
    
    if (currentPhase === 'PRE') {
        currentPhase = 'DURING';
        setupPhaseUI("Phase: DURING-Pranayama", [
            "Perform your pranayama exercise.",
            "Maintain stable face visibility.",
            "Deep, controlled breathing."
        ]);
        updateStatus("PRE session complete. Starting DURING session...");
    } else if (currentPhase === 'DURING') {
        currentPhase = 'POST';
        setupPhaseUI("Phase: POST-Session", [
            "Sit relaxed again.",
            "Keep eyes closed.",
            "Minimal movement.",
            "Measure recovery state."
        ]);
        updateStatus("DURING session complete. Starting POST session...");
    } else if (currentPhase === 'POST') {
        console.log("[TRANSITION] POST session complete, showing summary");
        showSummary();
        updateStatus("POST session complete. Generating final report...");
    } else {
        console.error("[TRANSITION] Unknown phase:", currentPhase);
        updateStatus("Error: Unknown phase transition");
    }
}

function formatTime(s) {
    const mins = Math.floor(s / 60);
    const secs = s % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// --- Phase Completion ---

function showPhaseComplete() {
    document.getElementById('processingStatus').style.display = 'none';
    document.getElementById('recordControls').style.display = 'block';
    
    const btn = document.getElementById('btnAction');
    btn.classList.remove('btn-stop');
    btn.classList.add('btn-start');
    
    document.getElementById(`step-${currentPhase.toLowerCase()}`).classList.add('completed');
    
    if (currentPhase === 'PRE') {
        btn.innerText = "Proceed to DURING Session";
    } else if (currentPhase === 'DURING') {
        btn.innerText = "Proceed to POST Session";
    } else {
        btn.innerText = "View Experiment Summary";
    }
}

function prepareNextPhase() {
    autoTransition();
}

function setupPhaseUI(label, instructions) {
    document.getElementById('currentPhaseLabel').innerText = label;
    const list = document.getElementById('instructionList');
    list.innerHTML = instructions.map(i => `<li>${i}</li>`).join('');
    
    document.getElementById('timerDisplay').innerText = "05:00";
    document.getElementById('progressBarFill').style.width = '0%';
    document.getElementById('btnAction').innerText = `Start Recording (${currentPhase})`;
    
// Orientation & Resize Handling
window.addEventListener('resize', () => {
    // If we have an active video stream, ensure ROI remains centered
    // (CSS handles most of this, but we can add JS checks if needed)
});

if (window.screen.orientation) {
    window.screen.orientation.onchange = () => {
        updateStatus(`Orientation changed: ${window.screen.orientation.type}`);
    };
}

function resetPhaseUI() {
    document.getElementById('processingStatus').style.display = 'none';
    document.getElementById('recordControls').style.display = 'block';
    document.getElementById('btnAction').innerText = `Retry Recording (${currentPhase})`;
}

// --- Summary & Dashboard ---

async function showSummary() {
    console.log("[SUMMARY] Starting summary display");
    try {
        switchPhase('summary');
        updateStatus("Experiment Complete. Reviewing results.");
        
        console.log("[SUMMARY] Populating comparison table and metric cards");
        populateComparisonTable();
        populateMetricCards();
        
        try {
            console.log("[SUMMARY] Fetching detailed results for:", experimentMetadata.subject.name, experimentMetadata.id);
            const response = await fetch(`/api/data-collection/results?name=${encodeURIComponent(experimentMetadata.subject.name)}&experiment_id=${encodeURIComponent(experimentMetadata.id)}`);
            const result = await response.json();
            console.log("[SUMMARY] Fetch response:", result);
            if (result.status === 'ok') {
                fullExperimentResults = result.results;
                renderUnifiedDashboard();
                switchSummaryTab('PRE');
            } else {
                console.log("[SUMMARY] Results fetch not ok, rendering comparison charts only");
                renderComparisonCharts();
            }
        } catch (err) {
            console.error("[SUMMARY] Failed to fetch detailed results:", err);
            console.log("[SUMMARY] Rendering comparison charts as fallback");
            renderComparisonCharts();
        }
        console.log("[SUMMARY] Display complete");
    } catch (err) {
        console.error("[SUMMARY] Fatal error:", err);
        alert("Error displaying summary: " + err.message);
    }
}

function populateMetricCards() {
    const phases = ['PRE', 'DURING', 'POST'];
    phases.forEach(p => {
        const res = phaseResults[p];
        if (res) {
            const bpmEl = document.getElementById(`val-${p.toLowerCase()}-bpm`);
            const rmssdEl = document.getElementById(`val-${p.toLowerCase()}-rmssd`);
            if (bpmEl) bpmEl.innerText = res.bpm || '--';
            if (rmssdEl) rmssdEl.innerText = (res.rmssd ? res.rmssd.toFixed(1) : '--') + ' ms';
        }
    });
}

function renderUnifiedDashboard() {
    renderComparisonCharts();
}

function renderComparisonCharts() {
    const labels = ['PRE', 'DURING', 'POST'];
    const bpmData = labels.map(l => phaseResults[l]?.bpm || 0);
    const rmssdData = labels.map(l => phaseResults[l]?.rmssd || 0);

    // BPM Comparison
    const ctxBpm = document.getElementById('bpmComparisonChart').getContext('2d');
    if (charts.bpmComp) charts.bpmComp.destroy();
    charts.bpmComp = new Chart(ctxBpm, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'BPM',
                data: bpmData,
                borderColor: '#45aaf2',
                backgroundColor: 'rgba(69, 170, 242, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: getChartOptions('Heart Rate (BPM)')
    });

    // HRV Comparison
    const ctxHrv = document.getElementById('hrvComparisonChart').getContext('2d');
    if (charts.hrvComp) charts.hrvComp.destroy();
    charts.hrvComp = new Chart(ctxHrv, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'RMSSD (ms)',
                data: rmssdData,
                backgroundColor: ['rgba(69, 170, 242, 0.6)', 'rgba(165, 94, 234, 0.6)', 'rgba(32, 191, 107, 0.6)'],
                borderRadius: 4
            }]
        },
        options: getChartOptions('HRV (RMSSD)')
    });
}

function switchSummaryTab(phase) {
    // Update UI
    document.querySelectorAll('.btn-tab').forEach(b => b.classList.remove('active'));
    document.getElementById(`tab-${phase}`).classList.add('active');
    
    if (!fullExperimentResults || !fullExperimentResults[phase]) {
        console.warn(`No detailed data for ${phase}`);
        return;
    }

    const data = fullExperimentResults[phase];
    
    // Update Waveform Chart
    const ctxWave = document.getElementById('tabWaveformCanvas').getContext('2d');
    if (charts.waveform) charts.waveform.destroy();
    charts.waveform = new Chart(ctxWave, {
        type: 'line',
        data: {
            labels: Array.from({length: data.filtered_signal.length}, (_, i) => i),
            datasets: [{
                label: `${phase} Waveform`,
                data: data.filtered_signal,
                borderColor: '#fff',
                borderWidth: 1,
                pointRadius: 0,
                tension: 0.1
            }]
        },
        options: getChartOptions('Filtered Signal (Normalised)', false)
    });

    // Update RR Interval Chart
    const ctxRR = document.getElementById('tabRRCanvas').getContext('2d');
    if (charts.rr) charts.rr.destroy();
    charts.rr = new Chart(ctxRR, {
        type: 'line',
        data: {
            labels: Array.from({length: data.rr_ms.length}, (_, i) => i + 1),
            datasets: [{
                label: 'RR Interval',
                data: data.rr_ms,
                borderColor: '#ffa502',
                backgroundColor: 'rgba(255, 165, 2, 0.1)',
                pointRadius: 2,
                fill: true
            }]
        },
        options: getChartOptions('RR Intervals (ms)')
    });
}

function getChartOptions(title, showX = true) {
    return {
        responsive: true,
        plugins: {
            legend: { display: false },
            title: { display: true, text: title, color: '#fff' }
        },
        scales: {
            y: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 10 } }
            },
            x: {
                display: showX,
                grid: { display: false },
                ticks: { color: 'rgba(255,255,255,0.5)', font: { size: 10 } }
            }
        }
    };
}

function renderBasicComparisonChart() {
    // Fallback if detailed results fail
    renderComparisonCharts();
}

function populateComparisonTable() {
    const tbody = document.getElementById('comparisonTableBody');
    const metrics = [
        { key: 'bpm', label: 'Heart Rate (BPM)' },
        { key: 'sdnn', label: 'SDNN (ms)' },
        { key: 'rmssd', label: 'RMSSD (ms)' },
        { key: 'pnn50', label: 'pNN50 (%)' },
        { key: 'mean_rr', label: 'Mean RR (ms)' }
    ];

    tbody.innerHTML = metrics.map(m => {
        const valPre = phaseResults.PRE ? phaseResults.PRE[m.key] : '--';
        const valDuring = phaseResults.DURING ? phaseResults.DURING[m.key] : '--';
        const valPost = phaseResults.POST ? phaseResults.POST[m.key] : '--';
        
        return `
            <tr>
                <td class="phase-name">${m.label}</td>
                <td style="color: #45aaf2;">${valPre}</td>
                <td style="color: #a55eea;">${valDuring}</td>
                <td style="color: #20bf6b;">${valPost}</td>
            </tr>
        `;
    }).join('');
}

    // (Removed old renderComparisonChart in favor of unified dashboard)
}

function downloadAllReports() {
    const name = experimentMetadata.subject.name;
    const expId = experimentMetadata.id;
    window.location.href = `/api/data-collection/export-all?name=${encodeURIComponent(name)}&experiment_id=${encodeURIComponent(expId)}`;
}

function updateStatus(msg) {
    document.getElementById('statusMessage').innerText = msg;
    document.getElementById('statusBadge').innerText = currentPhase;
}

function resumeExperiment() {
    // Explicit resume only.
    const saved = window.__rPPG_SAVED_STATE__;
    if (!saved) return;

    // Validate saved experiment is incomplete.
    if (!hasIncompleteExperimentSaved(saved)) {
        window.__rPPG_SAVED_STATE__ = null;
        return;
    }

    // Restore metadata only (not auto-advancing phase).
    experimentMetadata = saved.metadata || experimentMetadata;
    phaseResults = saved.phaseResults || phaseResults;

    // Re-enable resume UI button off.
    const resumeBtn = document.getElementById('btnResumeExperiment');
    if (resumeBtn) resumeBtn.disabled = true;

    // Keep UI at PRE until user progresses.
    switchPhase('record');
    currentPhase = 'PRE';
    try {
        document.getElementById('phase-info')?.classList.add('active');
        document.getElementById('phase-record')?.classList.remove('active');
        document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
        document.getElementById('step-info')?.classList.add('active');
    } catch {}

    // Fill subject fields.
    try {
        document.getElementById('subjectName').value = experimentMetadata.subject?.name || '';
        document.getElementById('experimentId').value = experimentMetadata.id || '';
        document.getElementById('subjectAge').value = experimentMetadata.subject?.age || '';
        document.getElementById('subjectGender').value = experimentMetadata.subject?.gender || '';
        document.getElementById('subjectNotes').value = experimentMetadata.subject?.notes || '';
    } catch {}

    updateStatus(`Recovered experiment ${experimentMetadata.id}. Continue from PRE.`);

    window.__rPPG_SAVED_STATE__ = null;
    localStorage.removeItem('rPPG_Experiment');
}

window.__rPPG_UI__maybeEnableResumeButton = function() {
    const saved = window.__rPPG_SAVED_STATE__;
    const btn = document.getElementById('btnResumeExperiment');
    if (!btn) return;
    btn.disabled = !saved;
};


// Hook up "Proceed" logic
document.addEventListener('click', (e) => {
    if (e.target.id === 'btnAction' && e.target.innerText.includes('Proceed')) {
        prepareNextPhase();
    } else if (e.target.id === 'btnAction' && e.target.innerText.includes('View Experiment Summary')) {
        showSummary();
    }
});

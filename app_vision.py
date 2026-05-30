# -*- coding: utf-8 -*-
"""
Panel Vision Inspection System
Vidana Consulting Pvt Ltd

Hardware:  NVIDIA GPU / CPU only
Model:     models/best.pt   (PyTorch YOLO)  → primary
Fallback:  models/best.pt   (PyTorch YOLO)  → CPU if GPU unavailable

Image capture per panel:
  SEQ1  →  full frame  +  left / middle / right / bottom crops
  SEQ2  →  full frame only
  SEQ3  →  full frame only
  SEQ4  →  full frame  +  serial-number ROI (rotated upright for OCR)

PDF generated automatically once all 4 sequences complete.
"""

import os
os.environ['ULTRALYTICS_OFFLINE'] = 'True'
os.environ['YOLO_VERBOSE'] = 'False'

import cv2
def speak(text, lang='en'):
    """
    Non-blocking TTS via espeak-ng / espeak.
    Used for: missed-sequence alerts and any other short messages.
    Silently skips if espeak is not installed.
    """
    import shutil
    exe = ('espeak-ng' if shutil.which('espeak-ng')
           else 'espeak' if shutil.which('espeak') else None)
    print(f'[TTS] [{lang}] {text}')
    if not exe:
        return
    def _run():
        try:
            subprocess.run(
                [exe, '-v', lang, '-s', '140', '-a', '90', text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=8)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def _announce_landscape(seq_num: int):
    """
    Speak landscape-placement instruction for SEQ2 or SEQ3.
    Plays English then Hindi.  Repeats once after 5 s if panel not yet
    captured (so operator doesn't miss it in a noisy factory).
    Non-blocking — runs entirely in a background thread.

    English : "Sequence 2. Please place the panel in landscape position"
    Hindi   : "Sequence do. Panel ko landscape position mein rakhein"
    """
    import shutil
    exe = ('espeak-ng' if shutil.which('espeak-ng')
           else 'espeak' if shutil.which('espeak') else None)

    seq_word_hi = {2: 'do', 3: 'teen'}.get(seq_num, str(seq_num))
    en_text = (f"Sequence {seq_num}. "
               f"Please place the panel in landscape position")
    hi_text = (f"Sequence {seq_word_hi}. "
               f"Panel ko landscape position mein rakhein")

    def _is_captured():
        if seq_num == 2:
            return getattr(state, 'seq2_auto_captured', False)
        if seq_num == 3:
            return getattr(state, 'seq3_auto_captured', False)
        return True

    def _play_once():
        if exe:
            try:
                subprocess.run(
                    [exe, '-v', 'en', '-s', '140', '-a', '90', en_text],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=8)
                time.sleep(0.3)
                subprocess.run(
                    [exe, '-v', 'hi', '-s', '120', '-a', '90', hi_text],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=8)
            except Exception as e:
                print(f'[TTS] espeak error: {e}')
        else:
            print(f'[TTS] EN: {en_text}')
            print(f'[TTS] HI: {hi_text}')

    def _run():
        print(f'[TTS] SEQ{seq_num} landscape announcement')
        _play_once()
        # Wait 5 s — repeat if operator hasn't placed panel yet
        time.sleep(5.0)
        if not _is_captured():
            print(f'[TTS] SEQ{seq_num} landscape reminder (repeat)')
            _play_once()

    threading.Thread(target=_run, daemon=True).start()

import numpy as np
import subprocess
import threading
import time
import select
import gc
import _queue

TEMP_DIR = "/home/vidana-pi/sspnew/temp_frames"
os.makedirs(TEMP_DIR, exist_ok=True)
import re
import platform
from datetime import datetime
from collections import deque

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from pdf_generator import generate_pdf_report
# import pytesseract  # Removed: Migrated to EasyOCR for memory efficiency on Pi

# Camera-2 OCR module (standalone — no Camera-1 dependency)
try:
    from camera2_ocr import Camera2OCR
    _CAM2_OCR_AVAILABLE = True
except ImportError:
    _CAM2_OCR_AVAILABLE = False
    print('[WARN] camera2_ocr.py not found — Camera-2 OCR disabled')

# [REMOVED] Tesseract path configuration — Migrated to EasyOCR
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Remove Flask's built-in size limit completely.
# We handle large files by streaming directly to disk (see upload_video).
app.config['MAX_CONTENT_LENGTH'] = None

# Return proper JSON on 413 just in case any proxy sends one
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# CONFIGURATION & PERFORMANCE
# ─────────────────────────────────────────────────────────────
MAX_PROC_WIDTH      = 320     # Lowered to 320 for maximum CPU speed (reduced lag)
FRAME_SKIP          = 2       # Process every 3rd frame
JPEG_QUALITY        = 40      # Fast encode
OCR_THROTTLE_FRAMES = 20      # OCR every 20 frames only
STREAM_FPS_CAP      = 12      # Reduced to 12 FPS for smoother performance on Pi
STREAM_UI_WIDTH      = 960     # Reduced from 1280 to 960 to save CPU during JPEG encoding

# ── Wiping completion — time-based ───────────────────────────
# How many seconds of continuous wiping = sequence complete
SEQ_WIPE_SECONDS    = 6.0    # 3s normal completion, 6s forced timeout
SEQ_HAND_GONE_SECS  = 1.0    # Seconds hand must be absent before SEQ3 can complete

# True PyTorch/Ultralytics class order from training
CLASS_NAMES = {
    0: 'hand',
    1: 'panel_seq1',
    2: 'panel_seq2',
    3: 'panel_seq3',
    4: 'serial_number',
}

# Reverse lookup: class name → model class index
CLASS_INDEX = {
    'hand':          0,
    'panel_seq1':    1,
    'panel_seq2':    2,
    'panel_seq3':    3,
    'serial_number': 4,
}

# panel class name → inspection sequence number (1-based)
CLASS_MAP = {
    "panel_seq1": 1,
    "panel_seq2": 2,
    "panel_seq3": 3,
}

# Human-readable short labels for SEQ panels
SEQ_DISPLAY_NAMES = {
    'panel_seq1': 'SEQ1',
    'panel_seq2': 'SEQ2',
    'panel_seq3': 'SEQ3',
}

DISPLAY_MAP = {
    "panel_seq1":           "SEQ1",
    "panel_seq2":           "SEQ2",
    "panel_seq3":           "SEQ3",
    "hand":                 "HAND",
    "serial_number":        "SERIAL",
    "serial_number_region": "SERIAL",
}

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(SCRIPT_DIR, "models")
PT_MODEL_PATH   = os.path.join(MODELS_DIR, "best.pt")
# [FIX] Set absolute storage path for Raspberry Pi production environment
BASE_STORAGE    = "/home/vidana-pi/SSP-SEQ/panel_data"
# Fallback for local testing if the Pi home directory is not available
if not os.path.exists("/home/vidana-pi"):
    BASE_STORAGE = os.path.join(SCRIPT_DIR, "SSP-SEQ")

for d in (MODELS_DIR, BASE_STORAGE):
    os.makedirs(d, exist_ok=True)



# ─────────────────────────────────────────────────────────────
# CPU INFERENCE WRAPPER
# ─────────────────────────────────────────────────────────────

class CPUInference:
    """YOLO inference — auto-selects CUDA GPU or CPU fallback."""
    def __init__(self, pt_path: str, force_cpu: bool = False):
        import torch
        from ultralytics import YOLO

        self._device = 'cpu'
        if not force_cpu and self._cuda_usable():
            self._device = 'cuda:0'

        self.model = YOLO(pt_path)
        self.model_names = getattr(self.model, 'names', {})

        try:
            self.model.predict(
                source=__import__('numpy').zeros((640, 640, 3), dtype='uint8'),
                device=self._device, verbose=False)
        except Exception as e:
            if self._device != 'cpu':
                print(f"  [WARN] CUDA warmup failed: {e}. Falling back to CPU.")
                self._device = 'cpu'
                self.model.predict(
                    source=__import__('numpy').zeros((640, 640, 3), dtype='uint8'),
                    device='cpu', verbose=False)
            else:
                raise

        mode = f'CUDA GPU ({self._device})' if self._device != 'cpu' else 'CPU'
        print(f"  ✅ YOLO ready — {pt_path}  [{mode}]")

    @staticmethod
    def _cuda_usable():
        import torch
        if not torch.cuda.is_available():
            return False
        try:
            torch.zeros(1, device='cuda:0')
            return True
        except Exception as e:
            print(f"  [WARN] CUDA available but unusable: {e}")
            return False

    def infer(self, frame: np.ndarray, conf_threshold: float = 0.25):
        results = self.model.predict(
            source=frame,
            device=self._device,
            conf=conf_threshold,
            verbose=False)
        
        # Raw debug (Every frame for first 100 frames, then every 30)
        f_idx = getattr(state, 'frame_count', 0)
        if f_idx < 100 or f_idx % 30 == 0:
            if results and len(results[0].boxes) > 0:
                dets = [(self.model_names.get(int(b.cls[0]), f"class_{int(b.cls[0])}"), float(b.conf[0]))
                        for b in results[0].boxes]
                print(f"  [AI DEBUG] Frame {f_idx} | Detections: {dets}")
            elif f_idx % 30 == 0:
                print(f"  [AI DEBUG] Frame {f_idx} | No boxes found")

        detections = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                name = self.model_names.get(class_id, CLASS_NAMES.get(class_id, f"class_{class_id}"))
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf,
                    'class_id': class_id,
                    'name': name,
                    'polygon': None
                })
        
        if f_idx % 30 == 0:
            print(f"[AI] Frame {f_idx} | Detections: {len(detections)}")
        return detections


# ─────────────────────────────────────────────────────────────
# ENGINE LOADER  — YOLO GPU / CPU only
# ─────────────────────────────────────────────────────────────

def load_inference_engine():
    """
    Smart priority — picks the FASTEST engine for the current hardware:

      Z440 / NVIDIA GPU  →  CUDA available           → GPU YOLO  (best.pt)
      No CUDA              → CPU YOLO  (best.pt)
    """
    import torch
    has_cuda = torch.cuda.is_available()
    device_name = 'N/A'
    if has_cuda:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception as e:
            device_name = f'unavailable ({e})'

    print("\n[INFO] Loading inference engine ...")
    print(f"[DEBUG] Active CLASS_NAMES: {CLASS_NAMES}")
    print(f"[INFO] CUDA available: {has_cuda}  ({device_name})")

    if os.path.exists(PT_MODEL_PATH):
        try:
            engine = CPUInference(PT_MODEL_PATH)
            mode = "GPU" if engine._device != 'cpu' else "CPU"
            print(f"[OK] Running on {mode} — YOLO best.pt\n")
            return engine, mode.lower()
        except Exception as e:
            print(f"  [ERR] YOLO load failed: {e}")
            if has_cuda:
                print("  [INFO] Retrying model load on CPU fallback...")
                try:
                    engine = CPUInference(PT_MODEL_PATH, force_cpu=True)
                    print("[OK] Running on CPU fallback — YOLO best.pt\n")
                    return engine, "cpu"
                except Exception as e2:
                    print(f"  [ERR] CPU fallback failed: {e2}")
    else:
        print(f"  [ERR] {PT_MODEL_PATH} not found")

    print("[ERR] No inference engine available — hand detection disabled\n")
    return None, "none"


# ─────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────

class PanelDetectionState:
    def __init__(self):
        self.camera_active        = False
        self.current_sequence = 0
        self.completed = {1: False, 2: False, 3: False}
        self.wipe_t0 = None
        self.progress = 0
        self.ocr_done = False
        self.serial = None
        self.last_clean = None
        self.orig = None
        from collections import deque
        self.seq_buf = deque(maxlen=5)
        self.wipe_buf = deque(maxlen=3)
        self.panel_id = 0
        self.status_msg = "IDLE"
        self.current_frame        = None
        self.panel_contour        = None
        self.panel_rect           = None
        self.panel_mask           = None
        self.cleaned_mask         = None
        self.current_sequence     = 0
        # status: 'active' | 'wiping' | 'completed' | 'pending' | 'missed'
        self.sequence_status      = {1: 'pending', 2: 'pending', 3: 'pending'}
        self.failure_reason       = None
        self.hand_bbox            = None
        self.prev_gray            = None
        self.motion_history       = deque(maxlen=10)
        self.panel_history        = deque(maxlen=5)
        self.last_panel_position  = None
        self.cleaning_percentage  = 0.0
        self.frame_count          = 0
        self.serial_number        = "UNKNOWN"
        self.sequence_captured    = {1: False, 2: False, 3: False}
        self.panel_lost_count     = 0       # Smoothing for overlays
        self.wiping_active        = False   # True when hand+motion detected
        self.current_panel_name   = None    # e.g., 'panel_seq1'
        self.ocr_frame_count      = 0
        self.serial_det           = None    # Storage for current model detection
        self.consecutive_wiping_frames = 0  # For smooth transitions
        self.seq3_absence_frames       = 0  # Counter for panel removal
        self.seq3_seen_frames          = 0  # Counter for panel presence
        self.seq1_wiping_frames   = 0
        self.seq2_wiping_frames   = 0
        self.seq3_wiping_frames   = 0
        self.no_wiping_frames     = 0       # For new panel detection
        self.panel_absent_frames  = 0       # Frames where no panel is detected
        self.ocr_candidates       = []      # Collection of OCR results for voting
        self.last_clean_frame        = None    # Best frame without hand for PDF
        self.last_clean_panel_rect   = None    # panel_rect at last clean frame
        self.last_clean_contour      = None    # contour at last clean frame
        self.last_clean_serial_det   = None    # serial det at last clean frame
        self.detected_seq_history = deque(maxlen=20) # Stronger debounce
        self.seq_capture_data     = {1: None, 2: None, 3: None}
        self.orig_frame           = None
        self.seq1_clean_images    = []
        self.max_ocr_images       = 5

        # ── Landscape alert (voice + screen colour for SEQ2/SEQ3) ────
        # Values: "" | "seq2" | "seq3" | "captured"
        self.landscape_alert        = ""
        self.landscape_alert_ts     = 0.0   # time alert was set
        self._seq2_cap_announced    = False  # True after "captured" voice played
        self._seq3_cap_announced    = False

        # ── Lead time tracking ────────────────────────────────────
        self.panel_start_time     = None          # when panel first detected
        self.seq_start_time       = {1: None, 2: None, 3: None}   # when each seq started
        self.seq_end_time         = {1: None, 2: None, 3: None}   # when each seq completed
        self.panel_end_time       = None          # when all seqs done
        self.current_sequence_panel_folder = None          # Locked folder for this panel
        self.panel_id             = None          # Unique ID for this panel session

        # ── Wipe time tracking (time-based completion) ────────────
        self.wipe_start_time      = None   # when continuous wiping started
        self.total_wipe_seconds   = 0.0    # accumulated wipe seconds this seq
        self.hand_gone_since      = None   # when hand last disappeared

        # ── SEQ3 completion tracking ──────────────────────────────
        # Track consecutive wiping/no-wiping frames for SEQ3
        self.seq3_wiping_frames   = 0      # Consecutive frames with wiping
        self.seq3_no_wipe_frames  = 0      # Consecutive frames without wiping after 5 wipes
        self.seq3_completion_stage = 0     # 0=waiting, 1=wiping, 2=checking_no_wipe

        # ── Auto-capture on sequence detection ──────────────────
        self.last_detected_seq    = 0      # Track which seq was last detected
        self.seq1_auto_captured   = False  # Has SEQ1 been auto-captured?
        self.seq2_auto_captured   = False  # Has SEQ2 been auto-captured?
        self.seq3_auto_captured   = False  # Has SEQ3 been auto-captured?
        # First unannotated frame the moment panel_seq1 appears
        self.seq1_first_clean_frame = None
        self.seq1_snapshot_data     = None   # Stores {'frame', 'rect', 'contour'} for SEQ1

        # ── Best landscape frame trackers (initialize at startup) ────────────
        # These store the sharpest HORIZONTAL frame seen for each sequence.
        self.seq2_best_frame      = None
        self.seq2_best_sharp      = 0.0
        self.seq2_best_meta       = {}
        self.seq2_landscape_count = 0      # consecutive landscape frames seen
        self.seq2_miss_count      = 0      # consecutive non-landscape frames
        self.seq2_prev_center     = None   # (cx, cy) for stability check

        self.seq3_best_frame      = None
        self.seq3_best_sharp      = 0.0
        self.seq3_best_meta       = {}
        self.seq3_landscape_count = 0
        self.seq3_miss_count      = 0
        self.seq3_prev_center     = None   # (cx, cy) for stability check
        self.seq3_absence_frames  = 0
        self.seq3_seen_frames     = 0

        # ── SEQ1 stability counter (direct access in process_frame) ──────────
        self.seq1_detection_count   = 0    # consecutive frames with panel+serial

        # ── Attributes set in reset_for_new_panel but missing from __init__ ──
        # All are accessed directly (not via getattr) in process_frame /
        # update_seq / update_wiping, so they MUST exist from construction.
        self.seq3_wiping_started    = False  # True once SEQ3 wiping begins
        self.was_absent_long        = False  # True if panel absent >30 frames
        self.ocr_triggered_by_tilt  = False  # True once portrait-tilt OCR fired
        self.seq2_stable_count      = 0      # stable-frame counter for SEQ2
        self.seq3_stable_count      = 0      # stable-frame counter for SEQ3
        self.seq2_in_sequence_frames = 0     # total frames spent in SEQ2
        self.audit_milestones       = set()  # progress milestones already captured
        # ── Wipe % captured at the moment of sequence transition (before reset) ──
        # Used by the SEQ1/SEQ2/SEQ3 completion gates to verify adequate wipe happened.
        self.seq1_final_wipe_pct    = 0      # SEQ1 progress% saved before SEQ2 reset
        self.seq2_final_wipe_pct    = 0      # SEQ2 progress% saved before SEQ3 reset
        self.seq2_final_wiping_frames = 0     # SEQ2 wiping frame count saved before reset
        self.seq2_detection_confidence = 0   # confidence of SEQ2 when first detected
        self.seq3_final_wipe_pct    = 0      # SEQ3 progress% saved before completion
        # FIX: panel_bbox_xyxy must be initialized — referenced by capture() via _meta()
        self.panel_bbox_xyxy        = None   # (x1, y1, x2, y2) of current panel bbox

        # ── Panel reset counter — increments every new panel ──────
        # Frontend watches this to know exactly when to reset bars
        self.panel_reset_id       = 0
        self.stream_generation    = 0
        self.local_camera_id      = 0
        self.current_sequence_panel_name = None
        self.last_wiping_time     = None
        self.model_sees           = 'NONE'

        # ── Daily panel counter ────────────────────────────────
        self.today_date           = datetime.now().strftime("%Y-%m-%d")
        self.panel_count_today    = self._load_daily_count()
        self.all_sequences_done   = False
        self._last_raw            = -1
        self._last_stable         = 0

        # ── Camera-2 OCR state (ADD ONLY — isolated from Camera-1) ────
        self.ocr_started          = False   # True once Camera2OCR.start_ocr() called
        self.partial_serial       = None    # live partial from cam2 OCR
        self.ocr_done             = False   # True once serial confirmed
        self.folder_renamed       = False   # True once folder is renamed to serial
        self.cam2_frame_saved     = False
        self.cam2_image_path      = None

        self.inference_engine, self.backend = load_inference_engine()

    def _load_daily_count(self):
        try:
            count_file = os.path.join(BASE_STORAGE, 'panel_count.txt')
            if os.path.exists(count_file):
                with open(count_file, 'r') as _f:
                    data = _f.read().strip().split(',')
                    if len(data) == 2 and data[0] == self.today_date:
                        return int(data[1])
        except Exception as e:
            print(f"[COUNT] Load error: {e}")
        return 0

    def refresh_daily_count(self):
        new_date = datetime.now().strftime("%Y-%m-%d")
        if new_date != self.today_date:
            self.today_date        = new_date
            self.panel_count_today = 0
        else:
            self.panel_count_today = self._load_daily_count()

    def reset_sequence_timers(self):
        """Reset wiping timers and masks when moving to a new sequence."""
        self.cleaned_mask         = None
        self.last_wiping_time     = None
        self.wipe_t0              = None
        self.progress             = 0
        self.cleaning_percentage  = 0.0
        self.wiping_active        = False
        self.no_wiping_frames     = 0
        self.consecutive_wiping_frames = 0
        self.wipe_start_time      = None
        self.total_wipe_seconds   = 0.0
        self.hand_gone_since      = None
        # Reset per-sequence wiping counters
        self.seq1_wiping_frames   = 0
        self.seq2_wiping_frames   = 0
        self.seq3_wiping_frames   = 0
        self.seq3_no_wipe_frames  = 0
        self.seq3_completion_stage = 0
        # ── Auto-capture flags: NEVER reset on sequence transitions ─────────
        # seq2_auto_captured / seq3_auto_captured must survive SEQ2→SEQ3 so
        # the fallback blocks don't re-fire after capture is already done.
        # Only reset_for_new_panel() may clear these.
        self.seq1_snapshot_data   = None   # clear so SEQ1 re-takes snapshot next panel
        self.seq3_absence_frames  = 0
        self.seq3_seen_frames     = 0
        # ── Landscape counters: reset per sequence (not best frames) ─────────
        # seq2_best_frame / seq3_best_frame are kept so rescue fallbacks work.
        # Only the per-sequence COUNTERS reset so the new sequence starts fresh.
        self.seq2_landscape_count = 0
        self.seq2_miss_count      = 0
        self.seq2_prev_center     = None
        self.seq3_landscape_count = 0
        self.seq3_miss_count      = 0
        self.seq3_prev_center     = None
        # FIX 1: clear audit_milestones on every sequence transition so
        # the 20/50/90% Camera-2 captures fire for EACH of SEQ1/SEQ2/SEQ3.
        # Without this, milestones captured in SEQ1 block SEQ2 and SEQ3 captures.
        self.audit_milestones     = set()
        # FIX 4: reset seq2_in_sequence_frames on sequence transition so the
        # 20-frame timeout does not fire immediately at SEQ2 start.
        self.seq2_in_sequence_frames = 0

    def reset_for_new_panel(self):

        """Reset all per-panel state when a new panel starts."""
        self.refresh_daily_count()
        self.current_sequence_seq = 0
        self.completed = {1: False, 2: False, 3: False}
        self.wipe_start = None
        self.progress = 0
        self.ocr_done = False
        self.serial = None
        self.current_sequence    = 0
        self.sequence_status     = {1: 'pending', 2: 'pending', 3: 'pending'}
        self.cleaned_mask        = None
        self.cleaning_percentage = 0.0
        self.failure_reason      = None
        self.last_panel_position = None
        self.serial_number       = "UNKNOWN"
        self.sequence_captured   = {1: False, 2: False, 3: False}
        self.panel_lost_count    = 0
        self.panel_absent_frames = 0
        self.wiping_active       = False
        self.ocr_candidates      = []
        self.all_sequences_done  = False
        self.panel_contour       = None
        self.panel_rect          = None
        self.panel_mask          = None
        self.hand_bbox           = None
        self.prev_gray           = None
        self.last_clean_frame      = None
        self.last_clean_panel_rect = None
        self.last_clean_contour    = None
        self.last_clean_serial_det = None
        self.seq_capture_data    = {1: None, 2: None, 3: None}
        self.detected_seq_history.clear()
        self.orig_frame = None
        self.seq1_clean_images.clear()
        self.seq1_snapshot_data = None
        self.seq3_absence_frames = 0
        self.seq3_seen_frames = 0
        # Reset lead times
        self.panel_start_time    = None
        self.seq_start_time      = {1: None, 2: None, 3: None}
        self.seq_end_time        = {1: None, 2: None, 3: None}
        self.panel_end_time      = None
        self.wipe_start_time     = None
        self.total_wipe_seconds  = 0.0
        self.hand_gone_since     = None
        self.last_wiping_time    = None
        self.current_sequence_panel_name = None
        self.current_sequence_panel_folder = None
        self.model_sees          = 'NONE'
        # Reset Camera-2 OCR flags for new panel
        self.ocr_started          = False
        self.ocr_done             = False
        self.partial_serial       = None
        self.folder_renamed       = False
        self.cam2_frame_saved     = False
        self.cam2_image_path      = None
        self.ocr_triggered_by_tilt = False   # reset portrait-tilt OCR flag
        # Reset SEQ3 completion tracking
        self.seq1_wiping_frames   = 0
        self.seq2_wiping_frames   = 0
        self.seq3_wiping_frames   = 0
        self.seq3_no_wipe_frames  = 0
        self.seq3_completion_stage = 0
        self.seq3_wiping_started  = False
        self.seq3_consec_wipe_frames = 0  # strict consecutive counter — must reset per panel
        self.seq1_final_wipe_pct  = 0    # saved at SEQ1→SEQ2 transition
        self.seq2_final_wipe_pct  = 0    # saved at SEQ2→SEQ3 transition
        self.seq2_final_wiping_frames = 0 # saved at SEQ2→SEQ3 transition
        self.seq2_detection_confidence = 0   # confidence of SEQ2 when first detected
        self.seq3_final_wipe_pct  = 0    # saved at SEQ3 completion
        self.current_sequence_panel_name = None
        self.last_wiping_time     = None
        # Reset auto-capture flags
        self.last_detected_seq      = 0
        self.seq1_auto_captured     = False
        self.seq2_auto_captured     = False
        self.seq3_auto_captured     = False
        # Reset stability counters for capture timing
        self.seq2_stable_count         = 0
        self.seq3_stable_count         = 0

        # Reset best landscape trackers for new panel
        self.seq2_best_frame      = None
        self.seq2_best_sharp      = 0.0
        self.seq2_best_meta       = {}
        self.seq2_landscape_count = 0
        self.seq2_miss_count      = 0
        self.seq2_prev_center     = None
        self.seq3_best_frame      = None
        self.seq3_best_sharp      = 0.0
        self.seq3_best_meta       = {}
        self.seq3_landscape_count = 0
        self.seq3_miss_count      = 0
        self.seq3_prev_center     = None
        self.seq3_absence_frames  = 0
        self.seq3_seen_frames     = 0
        # FIX H4: reset unconditional SEQ2 frame counter added by FIX H3
        self.seq2_in_sequence_frames   = 0
        # Reset saved wipe-% gates for the new panel
        self.seq2_final_wipe_pct  = 0
        self.seq3_final_wipe_pct  = 0
        # Clear first-frame snapshot so it is re-captured for each new panel
        self.seq1_first_clean_frame = None
        # FIX F1: reset attributes that are set dynamically and not listed above.
        # Without these, fallback captures use OLD panel images after a reset.
        self.seq1_detection_count   = 0    # counter for SEQ1 stability
        self.last_clean             = None # last hand-free frame (process_frame)
        self.orig                   = None # raw current frame (process_frame)
        self.was_absent_long        = False
        self.audit_milestones       = set()  # clear per-panel progress milestones
        self.panel_bbox_xyxy        = None   # FIX 3: reset per-panel panel bbox
        self.panel_id            = None   # DEFERRED — generated when first image captured
        _grabcut_cache.clear()
        self.panel_reset_id += 1
        # Reset landscape alert for new panel
        self.landscape_alert     = ""
        self.landscape_alert_ts  = 0.0
        self._seq2_cap_announced = False
        self._seq3_cap_announced = False
        print(f"🔄 New panel — all sequences reset (reset_id={self.panel_reset_id})")


state = PanelDetectionState()


# ─────────────────────────────────────────────────────────────
# PANEL DETECTION  (OpenCV — untouched)
# ─────────────────────────────────────────────────────────────

_grabcut_cache = {}   # {bbox_key: contour}

def detect_panel_contour_in_bbox(frame, bbox):
    """
    Returns a tight rectangular polygon based on the bbox.
    (Disabled GrabCut due to extreme CPU overhead on Raspberry Pi when wiping).
    """
    x1, y1, x2, y2 = bbox
    fh, fw = frame.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(fw, x2); y2 = min(fh, y2)
    
    return np.array([[[x1,y1]],[[x2,y1]],[[x2,y2]],[[x1,y2]]], dtype=np.int32)


# ─────────────────────────────────────────────────────────────
# HAND DETECTION  (calls active engine)
# ─────────────────────────────────────────────────────────────

def detect_hand(frame):
    """Refactored to generic detect_objects for all classes."""
    if state.inference_engine is None:
        return []
    try:
        # Lowered threshold to 0.15 for better sensitivity on CPU
        return state.inference_engine.infer(frame, conf_threshold=0.15)
    except Exception as e:
        print(f"Inference error: {e}")
    return []


# ─────────────────────────────────────────────────────────────
# MOTION DETECTION
# ─────────────────────────────────────────────────────────────

def detect_motion(frame, prev_gray):
    # Downscale to a small working resolution before any expensive operations.
    # At 1280×960, a 21×21 GaussianBlur takes ~100-150 ms on Pi 4B.
    # At 320×240 it takes ~6 ms — 25× faster.
    # detect_wiping() already resizes its inputs to (640,480) so the smaller
    # mask is fine: it just gets upscaled there instead of downscaled.
    MOTION_W, MOTION_H = 320, 240
    small = cv2.resize(frame, (MOTION_W, MOTION_H), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (9, 9), 0)   # smaller kernel for smaller frame
    if prev_gray is None:
        return gray, None
    # Shape mismatch guard: reconnect may change resolution mid-session
    if prev_gray.shape != gray.shape:
        return gray, None
    delta  = cv2.absdiff(prev_gray, gray)
    thresh = cv2.threshold(delta, 15, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    return gray, thresh


# ─────────────────────────────────────────────────────────────
# WIPING DETECTION  (small kernel — gradual progress)
# ─────────────────────────────────────────────────────────────

def detect_wiping(hand_bbox, motion_mask, panel_mask):
    """
    On CPU, motion might be jittery. If we have a hand inside the panel,
    we count it as wiping even if motion is subtle.
    """
    if hand_bbox is None or panel_mask is None:
        return None
    
    target_size = (640, 480)
    try:
        # Create hand mask
        hand_mask = np.zeros_like(panel_mask)
        x1, y1, x2, y2 = hand_bbox
        hand_mask[y1:y2, x1:x2] = 255
        
        # Resize both to 640x480
        p_resized = cv2.resize(panel_mask, target_size)
        h_resized = cv2.resize(hand_mask, target_size)
        
        # Core wiping: Intersection of Hand and Panel
        wiping = cv2.bitwise_and(h_resized, p_resized)
        
        # If motion mask exists, use it to refine
        if motion_mask is not None and motion_mask.size > 0:
            m_dilated = cv2.dilate(motion_mask, np.ones((15, 15), np.uint8))
            if m_dilated.shape[1] != target_size[0] or m_dilated.shape[0] != target_size[1]:
                m_dilated = cv2.resize(m_dilated, target_size)
            wiping = cv2.bitwise_and(wiping, m_dilated)
    except Exception as e:
        # print(f"[DEBUG] detect_wiping error: {e}")
        return 0.0
    
    # Expand the 'cleaned' area significantly around the hand
    wiping = cv2.dilate(wiping, np.ones((11, 11), np.uint8), iterations=1)
    return wiping


# ─────────────────────────────────────────────────────────────
# BLUE TINT
# ─────────────────────────────────────────────────────────────

def apply_blue_tint(frame, panel_mask, cleaned_mask):
    """Removed blue tint overlays: return the original frame unchanged.

    The panel mask/cleaned mask highlighting has been disabled per request.
    """
    return frame


# ─────────────────────────────────────────────────────────────
# CLEANING PERCENTAGE  (smoothed — max +2 % per frame)
# ─────────────────────────────────────────────────────────────

def calculate_cleaning_percentage(panel_mask, cleaned_mask):
    """Returns the raw % of the panel area covered by cleaned_mask.
    No per-frame cap — updates instantly so the progress bar reflects
    actual wipe coverage without lag.
    """
    if panel_mask is None or cleaned_mask is None:
        return 0.0
    panel_area   = cv2.countNonZero(panel_mask)
    if panel_area == 0:
        return 0.0
    cleaned_area = cv2.countNonZero(cv2.bitwise_and(panel_mask, cleaned_mask))
    return round((cleaned_area / panel_area) * 100.0, 1)


# ─────────────────────────────────────────────────────────────
# PANEL FLIP DETECTION
# ─────────────────────────────────────────────────────────────

def detect_panel_flip(is_panel_present, stable_model_seq):
    """
    Panel continuity logic:
    - Same panel if continuous or short gap, even if rotated.
    - Flip/New Panel ONLY if absent for > 60 frames AND a new panel (SEQ1) appears.
    """
    if not is_panel_present:
        state.panel_absent_frames += 1
        return False
    
    # Panel is present — reset absence counter
    absent = state.panel_absent_frames
    state.panel_absent_frames = 0
    
    # Protect against None (not enough history yet)
    if stable_model_seq is None:
        return False
    
    # If the panel was gone for > 60 frames (~2.4s at 25fps)
    # AND the model now detects a fresh SEQ1 panel, it's a completely new panel.
    # Otherwise, it's the same panel being rotated or repositioned.
    if absent > 60 and stable_model_seq == 1 and state.current_sequence > 0:
        print(f"🔄 New panel introduced (was absent for {absent} frames)")
        return True
        
    return False


# ─────────────────────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────────────────────

def crop_serial_roi_dynamic(frame, detections):
    for det in detections:
        if 'serial' in det['name'].lower():
            x1, y1, x2, y2 = det['bbox']

            # Add padding
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            return frame[y1:y2, x1:x2]

    return None

def crop_serial_roi_fallback(frame):
    h, w = frame.shape[:2]
    return frame[int(h*0.1):int(h*0.25), int(w*0.05):int(w*0.35)]

# OCR logic removed per request





# ─────────────────────────────────────────────────────────────
# FOLDER HELPER
# ─────────────────────────────────────────────────────────────

def get_panel_folder(serial_number, panel_id=None):
    """Returns a unique, stable folder path for the current panel session."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    if not panel_id:
        panel_id = datetime.now().strftime("%H%M%S")
    
    # Use a unique ID-based folder name instead of renaming based on serial
    # This prevents the "Missing Images" bug in PDF generation.
    folder_name = f"SSP-SEQ_{panel_id}"
    folder = os.path.join(BASE_STORAGE, date_str, folder_name)
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as e:
        if e.errno == 28:
            print(f"\n[CRITICAL] ❌ DISK FULL: Cannot create session folder. Please clear space on the device!")
        else:
            print(f"\n[ERROR] Folder creation failed: {e}")
        return None
    return folder


def ensure_panel_folder():
    """Returns the stable folder for the current panel session.

    Creates a new folder the first time it is called for a panel.
    Guards against creating a premature/orphaned folder by requiring
    that at least one sequence is active (current_sequence > 0) or
    that a panel_id is already set from this cycle.

    The orphaned-folder problem (SEQ2/SEQ3 detected before SEQ1) is
    prevented in update_seq() which blocks sequence advancement unless
    the previous sequence was genuinely active first.
    """
    # Between panels: no sequence active and no session open → return None
    # so callers skip rather than creating a premature folder.
    if (not state.panel_id          # no session open yet (None or 0)
            and state.current_sequence == 0
            and not state.seq1_auto_captured
            and not state.completed.get(1, False)):
        return state.current_sequence_panel_folder  # None

    if not state.panel_id:
        state.panel_id = datetime.now().strftime("%H%M%S")

    if state.current_sequence_panel_folder is None:
        state.current_sequence_panel_folder = get_panel_folder(
            state.serial_number, state.panel_id)

    return state.current_sequence_panel_folder



# ─────────────────────────────────────────────────────────────
# IMAGE CAPTURE
#
# SEQ1  →  full frame  +  left / middle / right / bottom crops
# SEQ2  →  full frame only
# SEQ3  →  full frame only
# SEQ4  →  full frame  +  ROI serial number (rotated 90° CCW)
# ─────────────────────────────────────────────────────────────

def rename_panel_images(folder, serial):
    """
    Rename all files saved with the placeholder prefix "PANEL" to the real
    serial number.  Recurses into the 'crops' subfolder too.
    Also renames OCR_Frame files so the serial is traceable.

    Before: PANEL_Seq1_Full_193145.jpg  /  OCR_Frame01_v1_193145.jpg
    After:  060526099A_Seq1_Full_193145.jpg  /  060526099A_OCR_Frame01_v1_193145.jpg
    """
    if not folder or not os.path.isdir(folder):
        print(f"  [RENAME] Skipped — invalid folder: {folder}")
        return

    if not serial or serial in ("Searching...", "UNKNOWN", "Reading...", "SSP-SEQ", ""):
        return

    safe_serial = re.sub(r'[^A-Za-z0-9_-]', '', str(serial))
    if not safe_serial:
        return

    import glob
    # Include crops subfolder
    search_dirs = [folder, os.path.join(folder, 'crops')]

    renamed_count = 0
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue

        # Rename SSP-SEQ_ prefixed files
        for old_path in glob.glob(os.path.join(search_dir, "SSP-SEQ_*.*")):
            basename = os.path.basename(old_path)
            new_basename = safe_serial + basename[len("SSP-SEQ"):]   # "SSP-SEQ_Seq1..." → "060526099A_Seq1..."
            new_path = os.path.join(search_dir, new_basename)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"  [RENAME] {basename} → {new_basename}")
                    renamed_count += 1
            except Exception as e:
                print(f"  [RENAME ERR] {basename}: {e}")

        # Rename legacy PANEL_ prefixed files
        for old_path in glob.glob(os.path.join(search_dir, "PANEL_*.*")):
            basename = os.path.basename(old_path)
            new_basename = safe_serial + basename[len("PANEL"):]   # "PANEL_Seq1..." → "060526099A_Seq1..."
            new_path = os.path.join(search_dir, new_basename)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"  [RENAME] {basename} → {new_basename}")
                    renamed_count += 1
            except Exception as e:
                print(f"  [RENAME ERR] {basename}: {e}")

        # Rename OCR_Frame files — add serial prefix so they are traceable
        for old_path in glob.glob(os.path.join(search_dir, "OCR_Frame*.jpg")):
            basename = os.path.basename(old_path)
            new_basename = f"{safe_serial}_{basename}"
            new_path = os.path.join(search_dir, new_basename)
            try:
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"  [RENAME] {basename} → {new_basename}")
                    renamed_count += 1
            except Exception as e:
                print(f"  [RENAME ERR] {basename}: {e}")

    if renamed_count:
        print(f"[RENAME] {renamed_count} files renamed to serial '{safe_serial}'")

def _save(path, img):
    """Saves image with high quality for PDF reports."""
    if img is None or img.size == 0:
        print(f"  ❌ Cannot save empty image to {os.path.basename(path)}")
        return False
    ok = cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"  {'💾' if ok else '❌'} {os.path.basename(path)}")
    return ok


def safe_serial_filename(serial):
    """
    Sanitize serial for use in filenames.
    ANY placeholder value → "SSP-SEQ" so rename_panel_images can find files by
    a single consistent prefix rather than trying to match 'Searching', 'UNKNOWN' etc.
    """
    _PLACEHOLDERS = {None, '', 'UNKNOWN', 'Searching...', 'Searching',
                     'Reading...', 'Reading', 'PANEL', 'SSP-SEQ', 'INVALID FRAME'}
    if serial in _PLACEHOLDERS:
        return "SSP-SEQ"
    s = re.sub(r'[^A-Za-z0-9_-]', '', str(serial))
    return s if s else "SSP-SEQ"



def capture_sequence_images(cap_data, sequence_id, serial_number):
    """
    Save images for a sequence at full camera resolution (JPEG quality=98).

    SEQ1 → Full landscape frame  +  Left / Middle / Right / Bottom crops
    SEQ2 → Full landscape frame only
    SEQ3 → Full landscape frame only

    CROP STRATEGY (SEQ1):
    Crops are taken from the FULL LANDSCAPE FRAME, NOT from the YOLO bbox.
    Reason: YOLO bbox can be portrait when panel is tilted → rotating the bbox
    crop makes Left/Middle/Right crops appear sideways.  The panel fills ~80%
    of the frame in the reference layout, so full-frame percentage crops give
    consistent, well-oriented views every time.

    Reference frame (CP Plus 1280×960 horizontal):
    ┌──────────────────────────────────────────┐
    │  Left  │     Middle      │    Right      │
    │  0-45% │   28%-72%       │  55%-100%     │
    │                                          │
    │        Bottom (55%-100% height)          │
    └──────────────────────────────────────────┘

    Filename prefix:
      Serial confirmed → {serial}_Seq{N}_Full_{ts}.jpg
      Serial pending   → PANEL_Seq{N}_Full_{ts}.jpg
    """
    try:
        folder = ensure_panel_folder()
        if folder is None:
            # Between-panel window: panel session not yet active.
            # Do NOT create a folder here — the capture is premature.
            print(f"  ❌ SEQ{sequence_id}: no active panel folder (between panels) — skipped")
            return None
        ts     = datetime.now().strftime("%H%M%S")
        safe_s = safe_serial_filename(serial_number)
        saved  = {}
        print(f"\n📸 Capturing SEQ{sequence_id} — serial='{safe_s}' → {folder}")

        frame = cap_data['frame']
        if frame is None or frame.size == 0:
            print(f"  ❌ SEQ{sequence_id}: frame is None/empty — skipped")
            return None

        fh, fw = frame.shape[:2]
        print(f"  Frame size: {fw}×{fh} px")

        # ── Step 1: Ensure landscape orientation ─────────────────────────
        # Camera is normally 1280×960 (landscape).  Guard against portrait
        # RTSP sources or cameras mounted on their side.
        # FIX D7: >= catches square frames (e.g. 640x640) too
        if fh >= fw:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            fh, fw = frame.shape[:2]
            print(f"  Rotated to landscape: {fw}x{fh}")

        # Report sharpness so we can track frame quality in logs
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # ── SEQ2/SEQ3: Always save the FULL horizontal camera frame ──────
        # STRICT: no bbox crop. Full landscape frame only.
        # Step 1 above already ensured landscape orientation.
        # A frame that is still portrait here means Step-1 failed (very unusual).
        if sequence_id in (2, 3):
            # Re-evaluate after possible rotation in Step 1
            if fh > fw:
                # Attempt a second rotation as last resort
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                fh, fw = frame.shape[:2]
                print(f"  ⚠️  SEQ{sequence_id}: secondary rotation applied → {fw}×{fh}")
            if fh > fw:
                print(f"  ❌ SEQ{sequence_id}: frame still NOT landscape "
                      f"({fw}×{fh}) after rotation — skipped")
                return None
            full_path = os.path.join(folder, f"{safe_s}_Seq{sequence_id}_Full_{ts}.jpg")
            if _save(full_path, frame):
                saved['main'] = full_path
                size_kb = os.path.getsize(full_path) // 1024
                print(f"  ✅ SEQ{sequence_id} FullFrame: {os.path.basename(full_path)}"
                      f"  ({fw}×{fh}, {size_kb} KB, sharpness={sharpness:.0f})")
            else:
                print(f"  ❌ SEQ{sequence_id} full-frame save FAILED")
                return None
            return saved

        # ── Step 2: Save full-resolution landscape frame (SEQ1 only) ─────
        full_path = os.path.join(folder, f"{safe_s}_Seq{sequence_id}_Full_{ts}.jpg")
        if _save(full_path, frame):
            saved['main'] = full_path
            size_kb = os.path.getsize(full_path) // 1024
            print(f"  ✅ SEQ{sequence_id} Full: {os.path.basename(full_path)}"
                  f"  ({fw}×{fh}, {size_kb} KB, sharpness={sharpness:.0f})")
        else:
            print(f"  ❌ SEQ{sequence_id} Full save FAILED")
            return None

        # ── Step 3: SEQ1 — four zone crops from the FULL FRAME ──────────
        if sequence_id == 1:
            crops_dir = os.path.join(folder, 'crops')
            os.makedirs(crops_dir, exist_ok=True)

            # ── Define crop zones as % of full frame ─────────────────────
            # Each crop spans the FULL HEIGHT so no panel features are cut.
            # Horizontal positions match the reference screenshot layout.
            #
            #   Left:   0% → 45%  width   (left edge + bracket area)
            #   Middle: 28% → 72% width   (centre openings and features)
            #   Right:  55% → 100% width  (right edge + holes)
            #   Bottom: 55% → 100% height (lower panel features + edge holes)
            #
            # Overlapping Left/Middle/Right ensures nothing is missed at
            # zone boundaries.

            x_l1, x_l2 = 0,              int(fw * 0.45)   # Left
            x_m1, x_m2 = int(fw * 0.28), int(fw * 0.72)   # Middle
            x_r1, x_r2 = int(fw * 0.55), fw               # Right
            y_b1, y_b2 = int(fh * 0.55), fh               # Bottom

            crop_defs = {
                'Left':   frame[:, x_l1:x_l2],
                'Middle': frame[:, x_m1:x_m2],
                'Right':  frame[:, x_r1:x_r2],
                'Bottom': frame[y_b1:y_b2, :],
            }

            for crop_name, img in crop_defs.items():
                if img is None or img.size == 0:
                    print(f"  ⚠️ Crop {crop_name} is empty — skipped")
                    continue
                ch, cw = img.shape[:2]
                if cw < 50 or ch < 50:
                    print(f"  ⚠️ Crop {crop_name} too small ({cw}×{ch}) — skipped")
                    continue
                crop_path = os.path.join(
                    crops_dir, f"{safe_s}_Seq1_{crop_name}_{ts}.jpg")
                if _save(crop_path, img):
                    saved[crop_name.lower()] = crop_path
                    print(f"  ✅ Crop {crop_name}: {cw}×{ch} px")

        return saved

    except Exception as e:
        import traceback
        print(f"❌ Capture error SEQ{sequence_id}: {e}")
        traceback.print_exc()
        return None


def _finalize_panel():
    """All 3 sequences done — increment count and fire PDF in background."""
    # [FIX] Capture the current folder BEFORE any state resets happen
    # This prevents the "PDF in next panel folder" bug.
    folder = state.current_sequence_panel_folder
    if folder is None:
        folder = ensure_panel_folder()
    
    if folder is None:
        print("⚠️ [PDF] Cannot generate PDF: No panel folder available (no images captured).")
        return

    if state.all_sequences_done:
        return
    state.all_sequences_done  = True
    state.panel_count_today  += 1
    # Persist panel count to file so it survives restarts
    try:
        count_file = os.path.join(BASE_STORAGE, 'panel_count.txt')
        os.makedirs(BASE_STORAGE, exist_ok=True)
        with open(count_file, 'w') as _f:
            _f.write(f"{state.today_date},{state.panel_count_today}")
    except Exception as _ce:
        print(f"[COUNT] Save error: {_ce}")
    state.current_sequence    = 0
    state.panel_end_time      = time.time()

    # Lock all statuses
    for i in (1, 2, 3):
        if state.sequence_status.get(i) not in ('completed', 'missed'):
            state.sequence_status[i] = 'completed'
            state.seq_end_time[i]    = state.panel_end_time

    # Calculate lead times
    panel_start = state.panel_start_time or state.panel_end_time
    total_secs  = state.panel_end_time - panel_start

    seq_times = {}
    for i in (1, 2, 3):
        s = state.seq_start_time.get(i)
        e = state.seq_end_time.get(i)
        if s and e:
            seq_times[i] = round(e - s, 1)
        else:
            seq_times[i] = None

    print("\n" + "="*60)
    print(f"🎉  PANEL {state.panel_count_today} COMPLETE — ALL SEQUENCES DONE")
    print(f"⏱️  TOTAL CUMULATIVE LEAD TIME: {total_secs:.1f}s")
    print("-" * 60)
    for i in (1, 2, 3):
        t = seq_times.get(i)
        print(f"    SEQ{i} Duration: {t:.1f}s" if t else f"    SEQ{i} Duration: N/A")
    print("="*60 + "\n")

    # Using the 'folder' path captured at the very beginning of finalization
    
    if state.serial_number:
        pdf_serial = state.serial_number
    else:
        pdf_serial = "UNKNOWN"
        
    serial_snap = pdf_serial
    seq_times_snap   = dict(seq_times)
    total_secs_snap  = total_secs
    panel_start_snap = panel_start

    if state.serial_number == "UNKNOWN":
        print("⚠️ Serial not detected properly")

    cam2_image = getattr(state, 'cam2_image_path', None)

    def _bg():
        try:
            # Snapshot reset_id at start — used to abort state writes if a
            # new panel cycle started (reset_panel was called) before this
            # PDF thread finishes. Without this guard, _bg() writing
            # state.serial_number after a reset overrides "Searching..." and
            # the UI shows the OLD panel's serial again (Bug 1).
            my_reset_id = state.panel_reset_id

            # ── FIX: wait up to 8s for background OCR to produce a real serial ──
            final_serial = serial_snap
            if _CAM2_OCR_AVAILABLE and camera2_ocr_instance is not None:
                t0 = time.time()
                while time.time() - t0 < 8.0:
                    s = camera2_ocr_instance.get_serial_number()
                    if s and s not in (None, 'UNKNOWN', 'Reading...', 'Searching...', ''):
                        final_serial = s
                        # Only update state if this panel hasn't been reset yet
                        if state.panel_reset_id == my_reset_id:
                            if state.serial_number in (None, 'UNKNOWN', 'Searching...', 'Reading...', ''):
                                state.serial_number = s
                        print(f"[PDF] OCR result ready in {time.time()-t0:.1f}s → serial={s}")
                        break
                    time.sleep(0.25)
                else:
                    print(f"[PDF] OCR not ready after 8s — using serial='{final_serial}'")
            # ── Rename all images with confirmed serial BEFORE generating PDF ─
            # This ensures every file in the folder carries the right serial
            # prefix and the PDF itself is stored in the serial-named folder.
            # FIX A1: always use the folder captured at _finalize_panel() entry.
            # state.current_sequence_panel_folder may point to the NEXT panel's
            # folder by the time _bg() runs (up to 8 s later for OCR wait).
            pdf_folder = folder
            if final_serial not in (None, 'UNKNOWN', 'Searching...', ''):
                rename_panel_images(pdf_folder, final_serial)
                print(f"[PDF] All images renamed → serial='{final_serial}'")

            # FIX F2: robustly locate the serial/cam2 image.
            # 1. Start with the early-captured local variable.
            # 2. Re-read state only if it still belongs to this panel (reset_id guard).
            # 3. Always fall back to a folder scan so reset_panel() can't lose the image.
            final_cam2_image = cam2_image  # early capture from _finalize_panel() start
            if state.panel_reset_id == my_reset_id:
                live = getattr(state, 'cam2_image_path', None)
                if live and os.path.exists(live):
                    final_cam2_image = live
            # Folder scan — finds Cam2_ROI, Cam2_FullFrame, Serial_Original, OCR_Frame
            if not final_cam2_image or not os.path.exists(str(final_cam2_image)):
                import glob as _gl
                for _pat in [
                    f"{final_serial}_Cam2_ROI_*.jpg",
                    f"{final_serial}_Cam2_FullFrame_*.jpg",
                    f"{final_serial}_Serial_Original.jpg",
                    "*_Cam2_ROI_*.jpg",
                    "*_Serial_Original.jpg",
                ]:
                    _hits = sorted(_gl.glob(os.path.join(pdf_folder, _pat)))
                    if _hits:
                        final_cam2_image = _hits[-1]
                        print(f'[PDF] Found serial image: {os.path.basename(final_cam2_image)}')
                        break
            if final_serial not in (None, 'UNKNOWN', 'Searching...', ''):
                import glob as _gl
                _ocr_hits = sorted(_gl.glob(
                    os.path.join(pdf_folder, f"{final_serial}_OCR_Frame*.jpg")))
                if _ocr_hits:
                    final_cam2_image = _ocr_hits[-1]

            print(f"[DEBUG PDF] 🚀 Generating PDF → {os.path.basename(pdf_folder) if pdf_folder else 'None'}")
            print(f"  - Serial: {final_serial}")
            print(f"  - Folder: {pdf_folder}")
            print(f"  - Cam2 Image: {final_cam2_image}")

            pdf_path = generate_pdf_report(
                final_serial,
                pdf_folder,
                seq_times=seq_times_snap,
                total_time=total_secs_snap,
                panel_start=panel_start_snap,
                cam2_image=final_cam2_image
            )
            if pdf_path:
                print(f"[DEBUG PDF] ✅ PDF generated successfully at: {pdf_path}")
            else:
                print("[DEBUG PDF] ⚠️  PDF generation returned None")
                state.failure_reason = "PDF Generation Failed: Missing images"
        except Exception as exc:
            import traceback
            err_msg = f"PDF Error: {str(exc)}"
            print(f"⚠️  {err_msg}")
            state.failure_reason = err_msg
            traceback.print_exc()

    threading.Thread(target=_bg, daemon=True).start()

    # Reset UI after a short delay so operator sees "COMPLETED" briefly.
    # Guard with reset_id snapshot: if a new panel arrives in <2s,
    # process_frame already called reset_panel() — don't reset again or
    # it will clear the new panel's serial mid-detection (Bug 3).
    _reset_id_at_finalize = state.panel_reset_id
    def _delayed_reset():
        time.sleep(2.0)
        if state.panel_reset_id == _reset_id_at_finalize:
            reset_panel()
        else:
            print("[RESET] Delayed reset skipped — new panel already started")
    threading.Thread(target=_delayed_reset, daemon=True).start()


def advance_sequence():
    """Manual advance for UI button"""
    if state.current_sequence < 3:
        update_seq(state.current_sequence + 1)
    else:
        complete(3)
    return True

    # SEQ3 done → full panel complete — route through _finalize_panel
    # which handles count increment + PDF + cleanup
    _finalize_panel()
    return False

def cleanup_panel_images(folder):
    """
    After PDF is confirmed saved, remove all .jpg images from the panel folder
    and its crops/ subfolder. Keeps the PDF. Never touches the _uploads folder.
    """
    import glob as _glob
    try:
        removed = 0
        # Remove JPGs in panel folder
        for jpg in _glob.glob(os.path.join(folder, '*.jpg')):
            try:
                os.remove(jpg)
                removed += 1
            except Exception:
                pass
        # Remove crops subfolder entirely
        crops_dir = os.path.join(folder, 'crops')
        if os.path.isdir(crops_dir):
            import shutil
            shutil.rmtree(crops_dir, ignore_errors=True)
        print(f"  [CLEANUP] Removed {removed} jpg(s) + crops/ from {folder}")
    except Exception as e:
        print(f"  [WARN] Cleanup failed: {e}")


# check_sequence_completion removed — completion is now driven
# exclusively by the sequence state machine in process_frame.


# ─────────────────────────────────────────────────────────────
# OVERLAY DRAWING
# ─────────────────────────────────────────────────────────────






# ─────────────────────────────────────────────────────────────
# SEQUENCE HELPERS
# ─────────────────────────────────────────────────────────────




# Duplicate sequence capture and finalization helpers removed

# ─────────────────────────────────────────────────────────────
# USER REQUESTED LOGIC
# ─────────────────────────────────────────────────────────────

def filter_detections(detections, frame):
    """
    Strict Filtering (User Requirement):
    1. Min confidence 0.6 (0.5 for hand)
    2. Only ONE detection per class
    3. Sequence-aware selection (Priority to current sequence)
    4. 70% Area Filter
    """
    # ── 5. DEBUG LOGGING (MANDATORY) ──
    print("\n--- MODEL DETECTIONS ---")
    if not detections:
        print("None")
    else:
        for d in detections:
            print(f"{d['name']} | conf={d['confidence']:.2f}")

    if not detections:
        return []
        
    best_per_class = {}
    for d in detections:
        name = d['name']
        conf = d.get('confidence', 0.0)
        
        # Normalization
        if name == "serial_number_region":
            name = "serial_number"
            d['name'] = "serial_number"
            
        # Confidence floors (Lowered hand threshold to 0.35 to be very aggressive)
        thresh = 0.35 if 'hand' in name else 0.6
        if conf < thresh:
            continue
            
        if name not in best_per_class or conf > best_per_class[name]['confidence']:
            best_per_class[name] = d
            
    # ── 2. Sequence-aware selection (STRICT) ──
    seq_candidates = [v for k, v in best_per_class.items() if k in CLASS_MAP]
    other_dets = [v for k, v in best_per_class.items() if k not in CLASS_MAP]
    
    # [RULE] Serial Number Presence = SEQ1
    has_serial = any(d['name'] == 'serial_number' for d in other_dets)
    
    panel_det = None
    if seq_candidates:
        # Expected ID: current active one, or 1 if just starting or if serial is detected
        expected_id = state.current_sequence
        if expected_id == 0:
            expected_id = 1
        
        # If we see a serial number, we MUST be looking for SEQ1
        if has_serial:
            expected_id = 1
            
        expected_name = f"panel_seq{expected_id}"

        # Priority 1: exact match with expected sequence
        for d in seq_candidates:
            if d['name'] == expected_name:
                panel_det = d
                break

        # Priority 2: fallback to highest confidence
        if panel_det is None:
            panel_det = max(seq_candidates, key=lambda x: x['confidence'])

    # ── 3. Strict sequence validation ──
    if panel_det is not None:
        det_id = CLASS_MAP.get(panel_det['name'], 0)
        
        # Determine what we expect (same logic as selection)
        expected_id = state.current_sequence if state.current_sequence > 0 else 1
        if has_serial:
            expected_id = 1
            
        if det_id != expected_id:
            print(f"[FILTER] Ignored wrong sequence {det_id}, expected {expected_id}")
            panel_det = None

    # ── 4. Area-based false detection filter ──
    if panel_det is not None:
        x1, y1, x2, y2 = panel_det['bbox']
        area = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]
        if area > 0.7 * frame_area:
            print("[FILTER] Ignored large false detection")
            panel_det = None
        
    return ([panel_det] if panel_det else []) + other_dets

def get_seq(dets):
    """
    Strict sequence selection: selects the HIGHEST confidence sequence detection.
    Guarantees no order-dependency.
    
    NOTE: If we are at the start (IDLE), we FAVOR SEQ1 if detected to prevent 
    model bias from skipping straight to SEQ2.
    """
    if not dets:
        state.model_sees = "NONE"
        return None

    # ONLY sequence detections (strict mapping)
    seq_dets = [d for d in dets if d['name'] in CLASS_MAP]

    if not seq_dets:
        state.model_sees = "NONE"
        return None

    # [FIX] If at start, prefer SEQ1 if present (avoids skipping)
    if state.current_sequence == 0:
        seq1 = next((d for d in seq_dets if CLASS_MAP[d['name']] == 1), None)
        if seq1:
            state.model_sees = seq1['name']
            return 1

    # Pick highest confidence ONLY (Standard rule)
    best = max(seq_dets, key=lambda x: x['confidence'])

    state.model_sees = best['name']
    return CLASS_MAP[best['name']]

def stable_seq(raw_seq):
    """
    Anti-flicker debounce for sequence detection.

    INITIAL detection (current_sequence == 0 → IDLE):
      - Needs only 2 consecutive frames of panel_seq1 before confirming.
      - Fast response — panel just arrived, no flicker risk yet.

    ACTIVE transitions (current_sequence 1→2, 2→3):
      - Needs 4 consecutive frames (was 6).
      - Still prevents false triggers from model flicker, but less delay.
      - At 5fps: 4 frames = 0.8s (was 1.2s with 6 frames).
    """
    val = raw_seq if raw_seq is not None else 0
    state.seq_buf.append(val)

    if val == getattr(state, '_last_raw', -1):
        state._consec_count = getattr(state, '_consec_count', 0) + 1
    else:
        state._consec_count = 1
    state._last_raw = val

    # Threshold depends on context
    if state.current_sequence == 0:
        # IDLE → SEQ1: 3 consecutive frames (fast response)
        threshold = 3
    else:
        # FIX I4: Active transitions (1→2, 2→3): reduced 4→3 consecutive frames.
        # At Pi ~3fps, weak SEQ3 detections (conf~0.39) may miss every 3rd frame;
        # 4-frame requirement means the transition never confirms.
        # 3 frames allows one missed frame while still preventing noise flicker.
        threshold = 3

    if state._consec_count >= threshold:
        state._last_stable = val

    return getattr(state, '_last_stable', 0)


# NOTE: is_wiping() removed — it was dead code (never called) but also
# mutated state.wipe_buf which is owned exclusively by update_wiping().

# OCR trigger functions removed

def capture(seq, frame, metadata=None):
    """
    Save images for a given sequence.
    Robust: guards against None frame and missing panel_rect.
    """
    if frame is None:
        print(f"[CAPTURE] SEQ{seq} skipped — frame is None")
        return None
    if frame.size == 0:
        print(f"[CAPTURE] SEQ{seq} skipped — frame is empty")
        return None

    _SERIAL_PLACEHOLDERS = (None, '', 'UNKNOWN', 'Searching...', 'Reading...')

    # SEQ2/SEQ3: wait up to 2s for Camera-2 OCR to confirm the serial so the
    # saved filename already carries the real serial (avoids SSP-SEQ prefix).
    if seq in (2, 3) and state.serial_number in _SERIAL_PLACEHOLDERS:
        if (getattr(state, 'ocr_started', False)
                and _CAM2_OCR_AVAILABLE
                and camera2_ocr_instance is not None):
            _t0 = time.time()
            while time.time() - _t0 < 2.0:
                _s = camera2_ocr_instance.get_serial_number()
                if _s and _s not in ('UNKNOWN', 'Reading...', 'Searching...', ''):
                    state.serial_number = _s
                    state.ocr_done = True
                    print(f"[CAPTURE] SEQ{seq} serial ready in "
                          f"{time.time()-_t0:.1f}s → {_s}")
                    break
                time.sleep(0.1)
            else:
                print(f"[CAPTURE] SEQ{seq}: serial not ready — "
                      "file saved as SSP-SEQ, renamed at finalization")

    serial = (state.serial_number
              if state.serial_number not in _SERIAL_PLACEHOLDERS
              else 'SSP-SEQ')

    # Use 'is not None' throughout — metadata values can be numpy arrays,
    # and numpy arrays raise ValueError if used with 'or' / bool() checks.
    def _meta(key, fallback_attr):
        v = metadata.get(key) if metadata is not None else None
        return v if v is not None else getattr(state, fallback_attr, None)

    cap_data = {
        'frame':         frame.copy(),
        'panel_rect':    _meta('rect',       'panel_rect'),
        'panel_contour': _meta('contour',    'panel_contour'),
        'serial_det':    _meta('serial_det', 'serial_det'),
        'bbox_xyxy':     _meta('bbox_xyxy',  'panel_bbox_xyxy'),
    }

    saved = capture_sequence_images(cap_data, seq, serial)
    if saved:
        state.sequence_captured[seq] = True
        print(f"[CAPTURE] ✅ SEQ{seq} images saved (serial={serial})")
    else:
        print(f"[CAPTURE] ⚠️ SEQ{seq} capture_sequence_images returned nothing")
    return saved

def complete(seq):
    if seq == 0 or state.completed.get(seq, False): return

    if seq in (2, 3):
        # Use state.sequence_captured[seq] as the authoritative "image saved" flag.
        # auto_captured can be True even when no image was saved (spam-prevention
        # path in fallbacks) — relying on it caused complete() to skip the rescue
        # even when the folder had no SEQ image at all.
        image_saved = state.sequence_captured.get(seq, False)
        if image_saved:
            print(f"[CAPTURE] SEQ{seq}: ✅ image confirmed saved")
        else:
            bf = (getattr(state, 'seq2_best_frame', None) if seq == 2
                  else getattr(state, 'seq3_best_frame', None))
            if bf is not None:
                print(f"[CAPTURE] SEQ{seq}: rescue via complete() — "
                      f"using landscape tracker best frame")
                result = capture(seq, bf,
                                 metadata=(getattr(state, 'seq2_best_meta', None) if seq == 2
                                           else getattr(state, 'seq3_best_meta', None)) or {})
                if result:
                    if seq == 2: state.seq2_auto_captured = True
                    else:        state.seq3_auto_captured = True
            else:
                # Last resort: use orig_frame or last_clean
                fb = state.orig_frame if state.orig_frame is not None else getattr(state, 'last_clean', None)
                if fb is not None:
                    fh, fw = fb.shape[:2]
                    if fw > fh:   # landscape only
                        print(f"[CAPTURE] SEQ{seq}: last-resort orig_frame rescue")
                        result = capture(seq, fb)
                        if result:
                            if seq == 2: state.seq2_auto_captured = True
                            else:        state.seq3_auto_captured = True
                    else:
                        print(f"[CAPTURE] SEQ{seq}: ⚠️ no valid frame for rescue — image skipped")
                else:
                    print(f"[CAPTURE] SEQ{seq}: ⚠️ no frame available — image skipped")
    else:
        best_frame = state.seq_capture_data.get(seq)
        if best_frame is None:
            best_frame = getattr(state, 'last_clean', None)
        if best_frame is None:
            best_frame = state.orig if state.orig is not None else None
        if best_frame is None:
            print(f"[CAPTURE] SEQ{seq}: no frame available — capture skipped")
        else:
            capture(seq, best_frame)

    state.completed[seq]         = True
    state.sequence_status[seq]   = 'completed'
    state.seq_end_time[seq]      = time.time()
    
    # Calculate duration if start time exists
    duration = 0
    if state.seq_start_time.get(seq):
        duration = state.seq_end_time[seq] - state.seq_start_time[seq]
        
    if seq < 3:
        state.status_msg = f"SEQ{seq} COMPLETED — FLIP PANEL TO SEQ{seq+1}"
    else:
        state.status_msg = f"SEQ{seq} COMPLETED"
        
    print(f"✅ [TRIGGERED] SEQ{seq} marked as COMPLETED | Duration: {duration:.1f}s")

    # [RULE] When SEQ1 is completed, do NOT immediately disable frame saving.
    # Interval frames are saved at 0s, 1.5s and 3.0s from scan start.
    # Disabling here causes frame-2 and frame-3 to be silently skipped if
    # SEQ1 completes in under 3 seconds (fast operator).
    # camera2_ocr.py self-manages the saving lifecycle — let it finish.
    # Also: only stop OCR scanning if OCR is already confirmed; otherwise the
    # OCR worker thread is mid-run and must not be interrupted.
    if seq == 1:
        if _CAM2_OCR_AVAILABLE and camera2_ocr_instance is not None:
            if camera2_ocr_instance.is_done():
                camera2_ocr_instance.stop_scanning()
                print('[CAM2-OCR] SEQ1 complete — OCR already done, scanning stopped')
            else:
                print('[CAM2-OCR] SEQ1 complete — OCR still running; '
                      'frame saving and scanning kept alive until serial confirmed')

    if seq == 3:
        print("\n" + "⭐" * 60)
        print("⭐  SEQ3 COMPLETION TRIGGERED — PANEL REMOVED AFTER WIPING  ⭐")
        print("⭐" * 60 + "\n")
        _finalize_panel()

def alert_missed(missed_seq):
    state.status_msg = f"SEQ{missed_seq} MISSED"
    speak(f"Sequence {missed_seq} missed")

def update_seq(new_seq):
    now = time.time()

    # IDLE — initialize ONLY when SEQ1 is detected (STRICT USER RULE)
    if state.current_sequence == 0:
        if new_seq == 1:
            state.current_sequence = 1
            state.sequence_status[1] = "active"
            state.status_msg = "SEQ1 ACTIVE"
            state.wipe_t0 = None
            state.progress = 0
            state.cleaning_percentage = 0.0
            state.cleaned_mask = None
            state.wiping_active = False
            state.seq3_wiping_frames = 0
            state.seq3_no_wipe_frames = 0
            state.seq3_completion_stage = 0
            if state.panel_start_time is None:
                state.panel_start_time = now
            if state.seq_start_time.get(new_seq) is None:
                state.seq_start_time[new_seq] = now
            state.current_sequence_panel_name = f"panel_seq{new_seq}"
            print(f"[PIPELINE] Initialized SEQ{new_seq}")
        else:
            # GUARD: block spurious SEQ2/SEQ3 detection before SEQ1 ever started.
            # This is the correct place to prevent orphaned folders — not in
            # ensure_panel_folder() which is called too late.
            print(f"[PIPELINE] ⚠️ Ignored SEQ{new_seq} — SEQ1 never started "
                  f"(current_seq=0). Spurious detection during placement?")
        return

    # Same sequence — no change
    if new_seq == state.current_sequence:
        return

    # FIX 5: always clear audit_milestones when transitioning sequences
    # so Camera-2 milestone captures (20/50/90%) fire fresh for each sequence.
    state.audit_milestones = set()

    # Higher sequence detected → operator moved to next area (only advance one step)
    if new_seq == state.current_sequence + 1:
        prev = state.current_sequence
        
        # [RULE] Sequence transitions: 
        # Lead times are now captured based on ACTUAL activity transitions 
        # handled in process_frame (e.g., SEQ2 only completes when SEQ3 wiping starts).
        state.current_sequence = new_seq
        state.sequence_status[new_seq] = "active"

        # ── Landscape voice + screen alert for SEQ2 and SEQ3 ────────
        if new_seq == 2:
            state.landscape_alert    = "seq2"
            state.landscape_alert_ts = time.time()
            _announce_landscape(2)
        elif new_seq == 3:
            state.landscape_alert    = "seq3"
            state.landscape_alert_ts = time.time()
            _announce_landscape(3)

        # ── SAVE final wipe % AND wiping frames for the PREVIOUS sequence ──
        # BEFORE progress/counters reset.  These saved values are the
        # authoritative records used by the SEQ1/SEQ2/SEQ3 completion gates.
        if prev == 1:
            state.seq1_final_wipe_pct = state.progress
            print(f'[SEQ1] 💾 Final wipe pct saved = {state.seq1_final_wipe_pct}% '
                  f'before progress reset for SEQ{new_seq}')
        elif prev == 2:
            state.seq2_final_wipe_pct = state.progress
            state.seq2_final_wiping_frames = state.seq2_wiping_frames
            print(f'[SEQ2] 💾 Final wipe pct saved = {state.seq2_final_wipe_pct}% '
                  f'wiping_frames saved = {state.seq2_final_wiping_frames} '
                  f'before progress reset for SEQ{new_seq}')
        elif prev == 3:
            state.seq3_final_wipe_pct = state.progress

        state.wipe_t0 = None
        state.progress = 0
        state.cleaning_percentage = 0.0
        state.cleaned_mask = None
        state.wiping_active = False
        
        # Reset counters for the new sequence
        state.seq1_wiping_frames = 0
        state.seq2_wiping_frames = 0
        state.seq3_wiping_frames = 0
        state.seq3_no_wipe_frames = 0
        state.seq3_completion_stage = 0
        state.seq3_wiping_started = False
        state.seq3_consec_wipe_frames = 0   # strict consecutive counter — reset on seq change
        
        state.status_msg = f"SEQ{new_seq} ACTIVE"
        if state.seq_start_time.get(new_seq) is None:
            state.seq_start_time[new_seq] = now
        state.current_sequence_panel_name = f"panel_seq{new_seq}"
        state.reset_sequence_timers()
        print(f"[PIPELINE] Moved to SEQ{new_seq}")

        # [RULE] When SEQ2 starts the panel has been flipped.
        # ONLY stop Camera-2 OCR if it has already confirmed the serial.
        # If OCR is still collecting frames / running OCR, keep it alive —
        # stopping early causes interval frames 2 & 3 and capture slots to be lost.
        if new_seq == 2:
            if _CAM2_OCR_AVAILABLE and camera2_ocr_instance is not None:
                if camera2_ocr_instance.is_done():
                    camera2_ocr_instance.stop_scanning()
                    print('[CAM2-OCR] ✅ OCR already confirmed — scanning stopped at SEQ2 start')
                else:
                    print('[CAM2-OCR] ⚠️  SEQ2 started but OCR NOT done — '
                          'keeping OCR alive to finish 3-slot capture + interval frames')

def update_wiping(hand_present, motion):
    # Wiping = hand present AND (motion detected OR hand overlaps panel)
    wiping_now = hand_present and (motion or (
        state.hand_bbox is not None and state.panel_mask is not None and
        state.panel_mask.shape[0] > 0))

    state.wipe_buf.append(1 if wiping_now else 0)
    wiping_confirmed = sum(state.wipe_buf) >= 2  # 2 out of 3 frames

    now = time.time()
    cur = state.current_sequence

    if wiping_confirmed:
        if state.wipe_t0 is None:
            state.wipe_t0 = now
            state.status_msg = f"SEQ{cur} WIPING"
            state.sequence_status[cur] = 'wiping'

        # Accumulate wipe time (only while actively wiping, gap < 2s allowed)
        if state.last_wiping_time is not None:
            elapsed = now - state.last_wiping_time
            if elapsed < 2.0:
                state.total_wipe_seconds = min(
                    SEQ_WIPE_SECONDS,
                    state.total_wipe_seconds + elapsed)
        state.last_wiping_time = now
        state.wiping_active = True

        # Progress calculation — use ACCUMULATED wipe time, NOT wall-clock time.
        # Wall-clock (now - wipe_t0) counted pauses as wiping time, letting
        # progress jump to 100% while the operator was idle.
        # total_wipe_seconds only grows while hand is actively on panel.
        # [RULE] For SEQ3, only progress if panel_seq3 is the active detection
        can_progress = True
        if cur == 3 and state.current_panel_name != "panel_seq3":
            can_progress = False

        if can_progress:
            state.progress = min(100, int(
                (state.total_wipe_seconds / SEQ_WIPE_SECONDS) * 100))
            state.cleaning_percentage = float(state.progress)

        # Increment per-sequence wiping counters
        if cur == 1:
            state.seq1_wiping_frames += 1

            # Trigger Camera-2 OCR at frame 3 (not 5) so all 3 capture slots
            # complete before the operator flips the panel.
            # At 3fps: frame 3 ≈ 1s of wiping. Slots at 0/1/2s need 2s total.
            # Triggering at 3 gives ~3s before a typical 4s flip = all slots fit.
            if state.seq1_wiping_frames >= 3 and not state.ocr_started:
                if state.seq1_snapshot_data is None:
                    # No real panel confirmed yet — gloves on empty table
                    print('[CAM2-OCR] ⚠️  SEQ1 snapshot not confirmed yet — '
                          'OCR trigger BLOCKED (empty table / no serial detected)')
                    # Fallback at frame 6 (was 8) — earlier to avoid missing slots
                    if state.seq1_wiping_frames >= 6:
                        folder = ensure_panel_folder()
                        if folder and camera2_ocr_instance is not None:
                            camera2_ocr_instance.set_panel_folder(folder)
                            camera2_ocr_instance.start_ocr()
                            state.ocr_started = True
                            print(f'[CAM2-OCR] ⚠️  Fallback OCR triggered at frame '
                                  f'{state.seq1_wiping_frames} (no snapshot yet — '
                                  f'serial class may be detected late)')
                            state.ocr_done = False
                            if not hasattr(state, 'audit_milestones'):
                                state.audit_milestones = set()
                            else:
                                state.audit_milestones.clear()
                else:
                    folder = ensure_panel_folder()
                    if folder and camera2_ocr_instance is not None:
                        camera2_ocr_instance.set_panel_folder(folder)
                        camera2_ocr_instance.start_ocr()
                        state.ocr_started = True
                        print(f'[CAM2-OCR] ✅ OCR triggered by SEQ1 wiping '
                              f'(frame {state.seq1_wiping_frames}, panel+serial confirmed)')

                        state.ocr_done = False
                        if not hasattr(state, 'audit_milestones'):
                            state.audit_milestones = set()
                        else:
                            state.audit_milestones.clear()

            # ── Trigger CAM2 Audit at Progress Milestones ────────────
            if camera2_ocr_instance is not None:
                for m in [20, 50, 90]:
                    if state.progress >= m and m not in getattr(state, 'audit_milestones', set()):
                        # Ensure cam2 panel_folder matches the current panel folder
                        # before saving the milestone frame.  If OCR was blocked
                        # (snapshot guard), panel_folder may still be "." which
                        # would save the image to the working directory and lose it.
                        _mf = ensure_panel_folder()
                        if _mf and camera2_ocr_instance.panel_folder != _mf:
                            camera2_ocr_instance.set_panel_folder(_mf)
                        print(f"[CAM2] Triggering {m}% milestone capture for SEQ{cur}")
                        camera2_ocr_instance.capture_single_audit_frame(m)
                        if not hasattr(state, 'audit_milestones'): state.audit_milestones = set()
                        state.audit_milestones.add(m)
                        # REMOVED: camera2_ocr_instance.ocr_done = True at 90%
                        # Forcing ocr_done prematurely killed serial detection:
                        # the OCR worker's _confirm_serial() checked `if self.ocr_done: return`
                        # and exited without saving the serial number.
        elif cur == 2:
            # [FIX] Count SEQ2 wipe frames whenever wiping is confirmed in SEQ2.
            # Previously gated on current_panel_name == "panel_seq2" — but during
            # the overlapping transition zone, the model sees BOTH panel_seq2 and
            # panel_seq3 simultaneously and may pick panel_seq3 as highest-conf,
            # causing the counter to stay at 0 even when the hand IS on the panel.
            # Wipe confirmation (hand_bbox + motion) is the real proof of wiping.
            _hand_on_panel = (
                state.hand_bbox is not None
                and state.panel_mask is not None
                and state.panel_mask.shape[0] > 0
            )
            if _hand_on_panel:
                state.seq2_wiping_frames += 1
                print(f"[SEQ2] Hand ON panel — wipe frame #{state.seq2_wiping_frames} "
                      f"(panel_name={state.current_panel_name})")
            
            # ── Trigger CAM2 Audit at Progress Milestones ────────────
            if camera2_ocr_instance is not None:
                for m in [20, 50, 90]:
                    if state.progress >= m and m not in getattr(state, 'audit_milestones', set()):
                        _mf = ensure_panel_folder()
                        if _mf and camera2_ocr_instance.panel_folder != _mf:
                            camera2_ocr_instance.set_panel_folder(_mf)
                        print(f"[CAM2] Triggering {m}% milestone capture for SEQ{cur}")
                        camera2_ocr_instance.capture_single_audit_frame(m)
                        if not hasattr(state, 'audit_milestones'): state.audit_milestones = set()
                        state.audit_milestones.add(m)

            # NOTE: seq2_in_sequence_frames is now incremented in process_frame
            # (after update_seq) so it counts ALL SEQ2 frames, not just wiping ones.

            # NOTE: seq_capture_data[2] is now managed by the landscape tracker
            # in process_frame (seq2_best_frame). Removed the old update here
            # that stored any hand-free frame regardless of orientation.

            # [RULE] SEQ2 ready to mark as completed
            if state.seq2_wiping_frames >= 6 and state.progress >= 80:
                print(f"[SEQ2] 6+ wiping frames and 80% progress — ready for completion")

            # [RULE] when wipe reaches 100%, make sure to capture the landscape frame, but do NOT complete SEQ2 here.
            # SEQ2 completion is strictly gated on the panel being flipped to SEQ3.
            if state.progress >= 100 and not state.completed.get(2, False):
                if not state.seq2_auto_captured:
                    fb = state.seq2_best_frame      # landscape tracked frame ONLY
                    # STRICT: SEQ2 must be horizontal — never fall back to last_clean/orig
                    if fb is not None:
                        capture(2, fb, metadata=state.seq2_best_meta or {})
                        state.seq2_auto_captured = True
                        state.seq_capture_data[2] = fb
                    else:
                        print("[SEQ2] ⚠️ 100% wipe but NO horizontal frame tracked — "
                              "waiting for landscape detection")
                # Save final wipe % so the process_frame gate sees a valid value
                state.seq2_final_wipe_pct = 100
                print("[SEQ2] 100% wipe reached — capture processed, waiting for panel flip to complete")
        elif cur == 3:
            # [FIX] Count SEQ3 wipe frames when hand is confirmed ON the panel.
            # The old gate (current_panel_name == "panel_seq3") was too strict:
            # during SEQ2→SEQ3 transition both panel classes appear together and
            # the model may still output panel_seq2, making the counter stuck at 0.
            # We still require the panel to be visible (panel_mask present) so
            # false positives from gloves on an empty table are blocked.
            # seq3_wiping_started is set only when panel_seq3 is the confirmed class
            # (unchanged) — this guards the SEQ3 wipe-% progress gate.
            _hand_on_panel_s3 = (
                state.hand_bbox is not None
                and state.panel_mask is not None
                and state.panel_mask.shape[0] > 0
            )
            if state.current_panel_name == "panel_seq3":
                state.seq3_wiping_started = True   # only set when SEQ3 class confirmed

            if _hand_on_panel_s3:
                state.seq3_wiping_frames += 1
                # Strict consecutive counter — resets to 0 on any non-wipe frame.
                # Used by SEQ2 completion gate: requires 10 CONTINUOUS wipe frames.
                state.seq3_consec_wipe_frames = \
                    getattr(state, 'seq3_consec_wipe_frames', 0) + 1
                print(f"[SEQ3] Hand ON panel — wipe frame #{state.seq3_wiping_frames} "
                      f"consec={state.seq3_consec_wipe_frames} "
                      f"(panel_name={state.current_panel_name})")
            state.seq3_no_wipe_frames = 0
            # NOTE: SEQ3 capture is handled EXCLUSIVELY by the landscape tracker.
            # Never store last_clean/state.orig here — those may be SEQ1/empty frames.

            # [RULE] SEQ3 completion handled when panel disappears in process_frame
            # but we need to reach 90% first (raised from 85%).
            if state.progress >= 90:
                print(f"[SEQ3] 90% reached — waiting for panel to fully disappear")

            # ── Trigger CAM2 Audit at Progress Milestones ────────────
            if camera2_ocr_instance is not None:
                for m in [20, 50, 90]:
                    if state.progress >= m and m not in getattr(state, 'audit_milestones', set()):
                        _mf = ensure_panel_folder()
                        if _mf and camera2_ocr_instance.panel_folder != _mf:
                            camera2_ocr_instance.set_panel_folder(_mf)
                        print(f"[CAM2] Triggering {m}% milestone capture for SEQ{cur}")
                        camera2_ocr_instance.capture_single_audit_frame(m)
                        if not hasattr(state, 'audit_milestones'): state.audit_milestones = set()
                        state.audit_milestones.add(m)

    else:
        state.last_wiping_time = None
        state.wiping_active    = False
        if hand_present:
            state.hand_gone_since = None
        elif state.hand_gone_since is None:
            state.hand_gone_since = now

        # SEQ3: reset wipe counter only if hand gone for long time
        if cur == 3:
            state.seq3_no_wipe_frames = getattr(state, 'seq3_no_wipe_frames', 0) + 1
            # Reset strict consecutive counter on ANY non-wipe frame
            state.seq3_consec_wipe_frames = 0
            if state.seq3_wiping_frames < 5 and state.seq3_no_wipe_frames >= 3:
                state.seq3_wiping_frames = 0
                state.seq3_no_wipe_frames = 0
                print(f"[SEQ3] Wiping interrupted — reset")

    # [RULE] Sequence completion logic has been moved to process_frame 
    # to support activity-based transitions (e.g. SEQ3 wiping confirms SEQ2 done).


def reset_panel():
    """Full system reset — called after PDF generation or when a new panel_seq1 arrives.
    Increments panel_reset_id ONCE so the frontend status poller fires resetAllProgressBars
    exactly once per panel cycle.
    """
    # Call reset_for_new_panel() which covers ALL fields including
    # _capture_slots, _slots_saved, intervals_saved that the old
    # piecemeal reset was missing — old scan data was bleeding into
    # the next panel and triggering stale OCR worker signals.
    global camera2_ocr_instance
    if camera2_ocr_instance is not None:
        try:
            camera2_ocr_instance.reset_for_new_panel()
        except Exception as e:
            print(f"[CAM2] reset_for_new_panel error: {e}")

    state.reset_for_new_panel()          # handles all per-panel state + increments panel_reset_id
    state.current_sequence = 0
    state.completed        = {1: False, 2: False, 3: False}
    state.all_sequences_done = False
    state.wipe_t0          = None
    state.progress         = 0
    state.ocr_done         = False
    state.serial           = None
    state.serial_number    = "Searching..."
    state.partial_serial   = None     # clear so UI hint resets cleanly
    state.seq_buf.clear()
    state.wipe_buf.clear()
    state._last_raw        = -1
    state._last_stable     = 0
    state._consec_count    = 0
    state.status_msg       = "READY (NEW PANEL)"
    
    # ── Camera-2 OCR reset (status message only — fields cleared above) ──
    if camera2_ocr_instance is not None:
        try:
            camera2_ocr_instance.status = "Waiting for new panel..."
        except Exception: pass
        
    print(f"[RESET] Panel reset complete — reset_id={state.panel_reset_id}")
    # Clear held annotation boxes so previous panel's bboxes don't bleed through
    with _overlay_lock:
        _held_panel.update( {'bbox': None, 'label': '', 'color': (200,200,200), 'ttl': 0})
        _held_hand.update(  {'bbox': None, 'ttl': 0})
        _held_serial.update({'bbox': None, 'ttl': 0})

# ─────────────────────────────────────────────────────────────────────────────
# STABLE STATE-DRIVEN OVERLAY RENDERING
# ─────────────────────────────────────────────────────────────────────────────
# Strategy: Never draw raw model predictions directly.
# Instead, maintain "held" boxes that survive for PANEL_HOLD_FRAMES after
# the last confirmed detection.  The panel box is driven by the state machine
# (confirmed sequence), not the noisy per-frame model output.
# This eliminates ALL bbox flicker regardless of backend (CPU or Hailo).
# ─────────────────────────────────────────────────────────────────────────────

# ── How many encoder frames to keep showing a box after the last detection ──
PANEL_HOLD_FRAMES = 15   # ~1 s at 15 FPS stream
HAND_HOLD_FRAMES  = 6    # ~0.4 s — hand boxes fade faster

# ── Per-class BGR colours ─────────────────────────────────────────────────────
_CLASS_COLORS = {
    'panel_seq1':           (0,   0, 230),    # Red
    'panel_seq2':           (0, 140, 255),    # Orange
    'panel_seq3':           (0, 210,  20),    # Green
    'hand':                 (0, 215, 255),    # Yellow
    'serial_number':        (255,  80,  80),  # Blue
}

# ── Human-readable display names ─────────────────────────────────────────────
_DISPLAY_NAME = {
    'panel_seq1':           'Panel  SEQ-1',
    'panel_seq2':           'Panel  SEQ-2',
    'panel_seq3':           'Panel  SEQ-3',
    'hand':                 'Hand',
    'serial_number':        'Serial No.',
    'serial_number_region': 'Serial No.',
}

# ── Rendering constants ───────────────────────────────────────────────────────
_FONT       = cv2.FONT_HERSHEY_DUPLEX
_FONT_SCALE = 0.70
_FONT_THICK = 2
_BOX_THICK  = 3   # slightly thicker for crisp look

# ── Held-box state (updated by infer thread, read by encoder thread) ─────────
# Each entry: {'bbox': (x1,y1,x2,y2), 'label': str, 'color': BGR, 'ttl': int}
_held_panel  = {'bbox': None, 'label': '', 'color': (200,200,200), 'ttl': 0}
_held_hand   = {'bbox': None, 'ttl': 0}
_held_serial = {'bbox': None, 'ttl': 0}
_overlay_lock = threading.Lock()


def _det_color(name):
    return _CLASS_COLORS.get(name, (200, 200, 200))


def _display(name):
    return _DISPLAY_NAME.get(name, name.replace('_', ' ').title())


def _draw_label(frame, text, x, y, color):
    """
    Near-black pill background  +  coloured border  +  white text.
    Positioned above the top-left corner of the bounding box.
    """
    (tw, th), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICK)
    pad  = 5
    bx1  = x
    by1  = max(0, y - th - pad * 2)
    bx2  = x + tw + pad * 2
    by2  = y
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
    cv2.putText(frame, text, (bx1 + pad + 1, by2 - pad + 1),
                _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICK + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (bx1 + pad, by2 - pad),
                _FONT, _FONT_SCALE, (255, 255, 255), _FONT_THICK, cv2.LINE_AA)


def update_held_boxes(detections, conf_seq):
    """
    Called by the INFER thread after process_frame().
    Updates the held-box state from the freshly confirmed detections.
    The panel box is driven by the STATE-MACHINE sequence (conf_seq),
    NOT by whichever raw class the model happened to output this frame.
    """
    global _held_panel, _held_hand, _held_serial

    panel_name = f'panel_seq{conf_seq}' if conf_seq in (1, 2, 3) else None
    color      = _CLASS_COLORS.get(panel_name, (200, 200, 200)) if panel_name else (200, 200, 200)

    # ── Panel box: pick the detection matching the confirmed sequence ────────
    panel_bbox = None
    best_conf  = 0.0
    detected_name = panel_name
    
    for d in detections:
        name = d.get('name', '')
        conf = d.get('confidence', 0.0)
        # Accept the detection only if it matches our confirmed sequence
        # (or any panel class when we are still at seq=0 / starting)
        if panel_name and name == panel_name and conf > best_conf:
            panel_bbox = d['bbox']
            best_conf  = conf
        elif panel_name is None and name.startswith('panel_seq') and conf > best_conf:
            panel_bbox = d['bbox']
            best_conf  = conf
            detected_name = name

    # If we are in the stabilization window (conf_seq=0) but we found a panel, use its color and name
    if panel_name is None and detected_name:
        color = _CLASS_COLORS.get(detected_name, (200, 200, 200))
        
    seq_label = _DISPLAY_NAME.get(detected_name, '') if detected_name else ''
    label = f"{seq_label}  {int(best_conf * 100)}%" if panel_bbox else seq_label

    # ── Hand box ─────────────────────────────────────────────────────────────
    hand_bbox = None
    hand_best = 0.0
    for d in detections:
        if d.get('name') == 'hand' and d.get('confidence', 0) > hand_best:
            hand_bbox = d['bbox']
            hand_best = d['confidence']

    # ── Serial box ───────────────────────────────────────────────────────────
    serial_bbox = None
    serial_best = 0.0
    for d in detections:
        if d.get('name') in ('serial_number', 'serial_number_region') \
                and d.get('confidence', 0) > serial_best:
            serial_bbox = d['bbox']
            serial_best = d['confidence']

    with _overlay_lock:
        # Update panel hold
        if panel_bbox is not None:
            _held_panel['bbox']  = panel_bbox
            _held_panel['label'] = label
            _held_panel['color'] = color
            _held_panel['ttl']   = PANEL_HOLD_FRAMES
        else:
            # Count down TTL so box fades after PANEL_HOLD_FRAMES encoder ticks
            if _held_panel['ttl'] > 0:
                _held_panel['ttl'] -= 1
            # Update label/color even when bbox is held (seq may have advanced)
            if panel_name:
                _held_panel['color'] = color
                _held_panel['label'] = seq_label  # no % when no live bbox

        # Update hand hold
        if hand_bbox is not None:
            _held_hand['bbox'] = hand_bbox
            _held_hand['ttl']  = HAND_HOLD_FRAMES
        else:
            if _held_hand['ttl'] > 0:
                _held_hand['ttl'] -= 1

        # Update serial hold
        if serial_bbox is not None:
            _held_serial['bbox'] = serial_bbox
            _held_serial['ttl']  = PANEL_HOLD_FRAMES
        else:
            if _held_serial['ttl'] > 0:
                _held_serial['ttl'] -= 1


def draw_overlay(frame, seq, status, progress):
    """
    State-driven overlay — draws held boxes, not raw model predictions.
    Boxes persist for PANEL_HOLD_FRAMES / HAND_HOLD_FRAMES frames after
    the last confirmed detection, so there is ZERO flicker.
    """
    with _overlay_lock:
        p_bbox  = _held_panel['bbox']  if _held_panel['ttl']  > 0 else None
        p_label = _held_panel['label']
        p_color = _held_panel['color']
        h_bbox  = _held_hand['bbox']   if _held_hand['ttl']   > 0 else None
        s_bbox  = _held_serial['bbox'] if _held_serial['ttl'] > 0 else None

    # ── Panel bounding box (one clean box, state-confirmed) ───────────────────
    if p_bbox is not None:
        try:
            x1, y1, x2, y2 = [int(v) for v in p_bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), p_color, _BOX_THICK)
            if p_label:
                _draw_label(frame, p_label, x1, y1, p_color)
        except Exception:
            pass

    # ── Hand bounding box ─────────────────────────────────────────────────────
    if h_bbox is not None:
        try:
            hx1, hy1, hx2, hy2 = [int(v) for v in h_bbox]
            hand_color = _CLASS_COLORS['hand']
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), hand_color, _BOX_THICK)
            _draw_label(frame, 'Hand', hx1, hy1, hand_color)
        except Exception:
            pass

    # ── Serial number bounding box ────────────────────────────────────────────
    if s_bbox is not None:
        try:
            sx1, sy1, sx2, sy2 = [int(v) for v in s_bbox]
            sn_color = _CLASS_COLORS['serial_number']
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), sn_color, _BOX_THICK)
            # Never display the actual serial number in the Camera-1 viewport
            _draw_label(frame, 'Serial No.', sx1, sy1, sn_color)
        except Exception:
            pass


    return frame



# ─────────────────────────────────────────────────────────────
# MAIN FRAME PROCESSING
# ─────────────────────────────────────────────────────────────

def process_frame(frame, detections):
    """Processes a single frame for the state machine."""
    if frame is None:
        return None

    h, w = frame.shape[:2]

    # ── 1. Filter detections ─────────────────────────────────────
    # FIX I1: TRACK A and TRACK B use DIFFERENT confidence thresholds.
    #
    # Production log showed panel_seq3 at conf=0.39 during SEQ2 wiping
    # (misclassification: model sees panel_seq1 on SEQ2 face at 0.94 conf).
    # With a shared 0.50 threshold, panel_seq3 was invisible to TRACK A so
    # the state machine never advanced to SEQ3 and no PDF was generated.
    #
    # TRACK A (state machine):  0.35 when panel active (allows weak
    #   cross-face transitions like panel_seq3 at 0.39 to advance state);
    #   0.50 when IDLE (prevents ghost badges on empty table).
    # TRACK B (captures only):  always 0.50 (ensures sharp, correct captures).
    _idle = (state.current_sequence == 0)
    _thresh_A = 0.60 if _idle else 0.35   # IDLE needs 0.60 (harder gate on empty table)
    _thresh_B = 0.50                       # captures

    seq_cands_A = [d for d in detections if d['name'].startswith('panel_seq')
                   and d['confidence'] >= _thresh_A]
    seq_cands_B = [d for d in detections if d['name'].startswith('panel_seq')
                   and d['confidence'] >= _thresh_B]
    hand_candidates = [d for d in detections if d['name'] == 'hand'
                       and d['confidence'] >= 0.5]
    serial_candidates = [d for d in detections
                         if d['name'] in ('serial_number', 'serial_number_region')
                         and d['confidence'] >= 0.3]

    # ── 2. Select BEST panel detection — TWO-TRACK logic ───────────
    #
    # TRACK A (raw_seq_id / best_any_panel):
    #   Low threshold — picks highest-confidence panel for state transitions.
    #   Allows weak SEQ3 detections (0.35+) to advance the state machine.
    #
    # TRACK B (panel_det):
    #   High threshold, class-locked — only used for capture decisions.
    #   Prevents wrong-face or blurry captures.

    # TRACK A — best detection of ANY panel class (drives state machine)
    best_any_panel = (max(seq_cands_A, key=lambda x: x['confidence'])
                      if seq_cands_A else None)
    raw_seq_id = 0
    if best_any_panel:
        try:
            raw_seq_id = int(best_any_panel['name'].replace('panel_seq', ''))
        except Exception:
            raw_seq_id = 0

    # ── GATE: SEQ1 from IDLE — panel size only ───────────────────────────────
    # best.hef detects: hand, panel_seq1, panel_seq2, panel_seq3.
    # It does NOT have a serial_number class — the serial gate was permanently
    # blocking SEQ1 (serial_for_seq1 was always False).
    # Protection against empty-table false positives is handled by:
    #   1. _thresh_A = 0.60 when IDLE (hard confidence gate)
    #   2. Panel bbox must cover ≥15% of the frame (size gate)
    if raw_seq_id == 1 and state.current_sequence == 0:
        panel_big_enough = False
        if best_any_panel:
            bx1, by1, bx2, by2 = best_any_panel['bbox']
            panel_px = max(0, bx2 - bx1) * max(0, by2 - by1)
            panel_big_enough = (panel_px >= 0.15 * h * w)

        if not panel_big_enough:
            # Panel bbox too small — likely a false positive
            raw_seq_id = 0
            state.seq1_detection_count = 0

    # TRACK B — class-locked detection (drives captures only)
    panel_det = None
    if seq_cands_B:
        expected_seq_id = state.current_sequence if state.current_sequence > 0 else 1
        expected_class  = f"panel_seq{expected_seq_id}"
        exact_matches   = [d for d in seq_cands_B if d['name'] == expected_class]
        if exact_matches:
            panel_det = max(exact_matches, key=lambda x: x['confidence'])
        else:
            # Log the mismatch (info only — state machine will handle transition)
            if best_any_panel:
                print(f"[FILTER] Capture locked to {expected_class} — "
                      f"seen '{best_any_panel['name']}' "
                      f"(conf={best_any_panel['confidence']:.2f}) [state machine ok]")


    # Best hand and serial
    hand_det   = max(hand_candidates,   key=lambda x: x['confidence']) \
                 if hand_candidates else None
    serial_det = max(serial_candidates, key=lambda x: x['confidence']) \
                 if serial_candidates else None

    # ── 4. Update shared state ────────────────────────────────────
    state.hand_bbox  = hand_det['bbox']  if hand_det   else None
    state.serial_det = serial_det        if serial_det else None
    hand_present     = state.hand_bbox is not None
    state.orig       = frame.copy()
    if not hand_present:
        state.last_clean = frame.copy()

    # [RULE] Trigger snapshot when panel_seq1 visible, no hand blocking.
    # has_serial removed — best.hef does not detect serial_number class.
    # Protection against empty table: _thresh_A=0.60 + panel size gate above.
    is_panel_seq1 = (raw_seq_id == 1)

    if is_panel_seq1 and not hand_present \
            and state.current_sequence in (0, 1):
        state.seq1_detection_count = getattr(state, 'seq1_detection_count', 0) + 1

        if state.seq1_detection_count == 1:
            state.status_msg = "SEQ1 Panel Detected — stabilising..."
            print(f"[SEQ1] Panel detected — waiting for 2 stable frames")

        # 2 consecutive clean frames → take snapshot
        should_capture = False
        if state.seq1_detection_count >= 2:
            should_capture = True
            print(f"[SEQ1] ✅ Panel stable for "
                  f"{state.seq1_detection_count} frames — taking snapshot")

        if should_capture and state.seq1_snapshot_data is None:
            frame_sharp = cv2.Laplacian(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

            panel_is_stable = True
            if frame_sharp < 12:  # FIX D2: 25->12 for Pi H.264 RTSP
                panel_is_stable = False
                print(f"[SEQ1] Frame too blurry (sharpness={frame_sharp:.0f} < 12) - waiting")

            if panel_is_stable:
                if panel_det is not None:
                    bx1, by1, bx2, by2 = panel_det['bbox']
                    # [RULE] Panel must be clearly horizontal (Landscape)
                    if (bx2 - bx1) < ((by2 - by1) * 1.05):  # FIX D2: 1.15->1.05 for Pi bbox rounding
                        print(f"[SEQ1] Panel is too vertical - waiting for x-axis alignment")
                        panel_is_stable = False
                    else:
                        current_rect = (bx1, by1, bx2 - bx1, by2 - by1)
                else:
                    # Fallback if only serial_number was seen: use full frame as panel_rect
                    current_rect = (0, 0, frame.shape[1], frame.shape[0])

            # Only commit snapshot when ALL stability checks pass
            if panel_is_stable:
                state.seq1_snapshot_data = {
                    'frame':      frame.copy(),
                    'rect':       current_rect,
                    'contour':    getattr(state, 'panel_contour', None),
                    'serial_det': serial_det,
                }
                state.seq1_first_clean_frame = state.seq1_snapshot_data['frame']
                # [NEW] Trigger folder creation exactly at snapshot moment
                ensure_panel_folder()
                print(f"[SEQ1] ✅ Snapshot taken and Folder Created at frame {state.seq1_detection_count} "
                      f"— sharpness={frame_sharp:.0f}")
    elif not is_panel_seq1:
        # Reset counter when panel is no longer visible
        if state.seq1_detection_count > 0:
            print(f"[SEQ1] Lost panel — resetting stability counter")
        state.seq1_detection_count = 0

    # ── PORTRAIT TILT → Camera-2 OCR trigger ─────────────────────────────────
    # When the operator tilts the panel from horizontal (landscape) to vertical
    # (portrait) during SEQ1, the serial-number face is now pointing at Camera-2.
    # Trigger Camera-2 OCR immediately at this moment — earlier than wiping.
    #
    # Conditions:
    #   • SEQ1 snapshot already taken  (real panel confirmed, not empty table)
    #   • best_any_panel visible with a portrait bbox  (height > width)
    #   • current_sequence == 1  (still in SEQ1 phase)
    #   • OCR not already started
    #   • Camera-2 instance available
    _panel_is_portrait = False
    if best_any_panel is not None:
        _bx1, _by1, _bx2, _by2 = best_any_panel['bbox']
        _bw = max(1, _bx2 - _bx1)
        _bh = max(1, _by2 - _by1)
        _panel_is_portrait = (_bh > _bw * 1.1)   # clearly taller than wide

    if (_panel_is_portrait
            and state.current_sequence == 1
            and state.seq1_snapshot_data is not None
            and not getattr(state, 'ocr_started', False)
            and _CAM2_OCR_AVAILABLE
            and camera2_ocr_instance is not None):
        folder = ensure_panel_folder()
        if folder:
            camera2_ocr_instance.set_panel_folder(folder)
            camera2_ocr_instance.start_ocr()
            state.ocr_started = True
            print('[CAM2-OCR] 🔄 OCR triggered by panel PORTRAIT TILT '
                  '(serial face now pointing at Camera-2)')

    # ── 5. Update panel contour + mask from detection ─────────────
    if panel_det is not None:
        bx1, by1, bx2, by2 = panel_det['bbox']
        pad_x = int((bx2 - bx1) * 0.03)
        pad_y = int((by2 - by1) * 0.03)
        bx1 = max(0, bx1 - pad_x)
        by1 = max(0, by1 - pad_y)
        bx2 = min(w, bx2 + pad_x)
        by2 = min(h, by2 + pad_y)
        state.panel_contour  = detect_panel_contour_in_bbox(frame, (bx1, by1, bx2, by2))
        state.panel_rect     = (bx1, by1, bx2 - bx1, by2 - by1)
        state.panel_bbox_xyxy = (bx1, by1, bx2, by2)   # FIX 3: update for capture()
        pm = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(pm, [state.panel_contour], -1, 255, -1)
        state.panel_mask = pm
        state.current_panel_name = panel_det['name']
    elif best_any_panel is not None:
        # FIX 2: TRACK B (panel_det) may be None due to class-lock during transitions.
        # Update current_panel_name from TRACK A (best_any_panel) as fallback so
        # seq2_wiping_frames / seq3_wiping_frames counters are not blocked by a
        # stale class name from the previous sequence.
        _bap_name = best_any_panel['name']
        _expected  = f'panel_seq{state.current_sequence}'
        if _bap_name == _expected:
            state.current_panel_name = _bap_name
    else:
        pass  # panel absent — current_panel_name left as-is for hold-decay logic
        
    if best_any_panel is not None:
        if getattr(state, 'panel_absent_frames', 0) > 30:
            state.was_absent_long = True
        state.panel_absent_frames = 0
    else:
        state.panel_absent_frames = getattr(state, 'panel_absent_frames', 0) + 1
        # FIX B2: force held overlay off after 8 absent frames so UI badge
        # clears immediately when panel is physically removed on Pi.
        if state.panel_absent_frames > 8:
            with _overlay_lock:
                _held_panel['ttl'] = 0
            state.model_sees = 'NONE'
    # ── 6. Stable sequence buffer ─────────────────────────────────
    conf_seq = stable_seq(raw_seq_id)

    # ── SEQ1 ─────────────────────────────────────────────────────────
    # Guard: only attempt once. If capture() fails (disk full, folder race etc.)
    # we do NOT retry every frame at 25fps — that floods logs and wastes I/O.
    # seq1_auto_captured is set True on the first attempt regardless of outcome;
    # a failure is logged clearly so the operator can investigate.
    if not state.seq1_auto_captured and state.seq1_snapshot_data is not None:
        state.seq1_auto_captured = True   # mark immediately — no repeated attempts
        ok = capture(1, state.seq1_snapshot_data['frame'], metadata=state.seq1_snapshot_data)
        if ok:
            print("[CAPTURE] SEQ1 ✅ — full frame + crops saved")
        else:
            print("[CAPTURE] SEQ1 ❌ — capture_sequence_images failed "
                  "(check folder, disk space, frame validity)")


    # ── SEQ2  ─────────────────────────────────────────────────────────────────
    # Strategy: track the BEST sharpness horizontal frame during SEQ2.
    # Capture fires once we have 3 good landscape frames — using the sharpest.
    #
    # Detection source priority:
    #   1. panel_det  (TRACK B — conf≥0.50, class-locked to panel_seq2)
    #   2. best_any_panel  (TRACK A — conf≥0.35, used when TRACK B is None)
    # Using TRACK A as fallback prevents missed captures when model outputs
    # panel_seq2 at 0.40-0.49 conf (common on Pi with RTSP compression).
    #
    # Stable counter:
    #   • Increments on every valid landscape frame
    #   • Only resets after 3 CONSECUTIVE non-landscape / no-detection frames
    #   • This tolerates 1-2 frame detection gaps without losing progress
    seq3_also_visible = any(d['name'] == 'panel_seq3' for d in seq_cands_A)

    # Best available panel_seq2 detection this frame
    _seq2_det = None
    if panel_det is not None and panel_det['name'] == 'panel_seq2':
        _seq2_det = panel_det
    elif (best_any_panel is not None
          and best_any_panel['name'] == 'panel_seq2'
          and not seq3_also_visible):
        _seq2_det = best_any_panel

    if (state.current_sequence == 2
            and not state.seq2_auto_captured
            and not state.all_sequences_done):

        if _seq2_det is not None:
            bx1, by1, bx2, by2 = _seq2_det['bbox']
            _bw = max(1, bx2 - bx1)
            _bh = max(1, by2 - by1)
            _det_conf   = _seq2_det['confidence']
            _panel_area = _bw * _bh
            # PANEL-BBOX landscape check: panel must be wider than tall.
            # Frame is always 1280x960 (w>h always True) so using
            # frame dims never rejected portrait panels — fixed.
            _is_landscape = _bw > _bh * 1.1
            _is_big_enough = (_panel_area >= 0.15 * w * h)
            _cx = (bx1 + bx2) // 2
            _cy = (by1 + by2) // 2
            _prev = getattr(state, 'seq2_prev_center', None)
            _is_stable = True
            if _prev is not None:
                _dx = abs(_cx - _prev[0])
                _dy = abs(_cy - _prev[1])
                _is_stable = _dx < 20 and _dy < 20
            state.seq2_prev_center = (_cx, _cy)

            if _is_landscape and _is_big_enough:
                frame_sharp = cv2.Laplacian(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    cv2.CV_64F).var()

                if frame_sharp >= 12:
                    # ── Layer 1: Always track best frame (even with hand) ────────
                    # Ensures hand/timeout fallbacks always have a frame to use.
                    if _det_conf >= 0.40 and frame_sharp > state.seq2_best_sharp:
                        state.seq2_best_frame = frame.copy()
                        state.seq2_best_sharp = frame_sharp
                        state.seq2_best_meta  = {
                            'rect':      (bx1, by1, _bw, _bh),
                            'bbox_xyxy': (bx1, by1, bx2, by2),
                            'contour':   getattr(state, 'panel_contour', None),
                            'serial_number': state.serial_number,
                        }
                        _htag = " (with hand)" if hand_present else ""
                        print(f"[SEQ2] Best frame updated{_htag} "
                              f"sharp={frame_sharp:.0f} conf={_det_conf:.2f}")

                    # ── Layer 2: Fire capture — no hand + stable required ────────
                    if not hand_present and _is_stable:
                        state.seq2_miss_count = 0
                        state.seq2_landscape_count += 1
                        print(f"[SEQ2] Landscape #{state.seq2_landscape_count} "
                              f"sharp={frame_sharp:.0f} conf={_det_conf:.2f} "
                              f"area={_panel_area}/{int(0.15*w*h)}")

                        _fire_threshold = (1 if (_det_conf >= 0.50 and frame_sharp >= 15
                                                 and state.seq2_best_frame is not None)
                                           else 3)
                        if state.seq2_landscape_count >= _fire_threshold:
                            best_f = state.seq2_best_frame or frame.copy()
                            print(f"[CAPTURE] SEQ2 ✅ confirmed "
                                  f"({state.seq2_landscape_count} frame(s), "
                                  f"sharp={state.seq2_best_sharp:.0f})")
                            capture(2, best_f, metadata=state.seq2_best_meta)
                            state.seq2_auto_captured = True
                            state.seq_capture_data[2] = best_f
                else:
                    print(f"[SEQ2] Landscape but blurry ({frame_sharp:.0f})")

            elif not _is_big_enough:
                state.seq2_miss_count = getattr(state, 'seq2_miss_count', 0) + 1
                print(f"[SEQ2] Panel too small "
                      f"({_panel_area} < {int(0.15*w*h)}) — skipping")
            elif _is_landscape and not _is_stable:
                state.seq2_landscape_count = 0
                print(f"[SEQ2] MOVING — waiting for stable position")
            else:
                state.seq2_miss_count = getattr(state, 'seq2_miss_count', 0) + 1
                if state.seq2_miss_count >= 8:
                    state.seq2_landscape_count = 0
                    state.seq2_miss_count = 0
                    print(f"[SEQ2] 8 miss frames — resetting counter ({w}\u00d7{h})")
        else:
            state.seq2_miss_count = getattr(state, 'seq2_miss_count', 0) + 1
            if state.seq2_miss_count >= 8:
                state.seq2_landscape_count = 0
                state.seq2_miss_count = 0

    # ── SEQ2 fallback: transition to SEQ3 happened but SEQ2 not captured ──────
    # STRICT: Only use tracked landscape frame — never fall back to last_clean/orig.
    if (conf_seq == 3 and state.current_sequence == 3
            and not state.seq2_auto_captured
            and not state.all_sequences_done):
        fb = state.seq2_best_frame      # landscape tracked frame ONLY
        if fb is not None:
            print("[CAPTURE] SEQ2 FALLBACK — using tracked horizontal frame")
            capture(2, fb, metadata=state.seq2_best_meta or {})
            state.seq2_auto_captured = True
            state.seq_capture_data[2] = fb
        else:
            print("[CAPTURE] SEQ2 FALLBACK ⚠️ no frame tracked — "
                  "will retry via timeout or complete() rescue")

    # ── CROSS-SEQUENCE OPPORTUNISTIC CAPTURE ────────────────────────────────
    # When the model detects the OTHER sequence's panel class at high confidence
    # with a full landscape frame, save it immediately — irrespective of which
    # sequence is currently active.
    #
    # Use-case A: SEQ2 active → model briefly detects panel_seq3 alone (clean
    #             horizontal view during flip). Save it now rather than waiting.
    # Use-case B: SEQ3 active → model detects panel_seq2 alone (operator
    #             momentarily reveals SEQ2 face). Save it if not yet captured.
    #
    # Guards: conf >= 0.55, landscape, big enough, not already captured.
    _cross_det = best_any_panel
    if _cross_det is not None and not hand_present:
        _cross_conf = _cross_det.get('confidence', 0)
        _cross_name = _cross_det.get('name', '')
        _cbx1, _cby1, _cbx2, _cby2 = _cross_det['bbox']
        _cw = max(1, _cbx2 - _cbx1); _ch = max(1, _cby2 - _cby1)
        _c_landscape  = w > h
        _c_big_enough = (_cw * _ch) >= (0.15 * w * h)

        if _cross_conf >= 0.55 and _c_landscape and _c_big_enough:
            _c_sharp = cv2.Laplacian(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

            # SEQ2 active but SEQ3 class detected alone → pre-capture SEQ3
            if (state.current_sequence == 2 and _cross_name == 'panel_seq3'
                    and not state.seq3_auto_captured
                    and not state.all_sequences_done
                    and _c_sharp >= 12):
                if _cross_conf > getattr(state, 'seq3_best_sharp', -1):
                    state.seq3_best_frame = frame.copy()
                    state.seq3_best_sharp = _cross_conf
                    state.seq3_best_meta  = {
                        'rect':      (_cbx1, _cby1, _cw, _ch),
                        'bbox_xyxy': (_cbx1, _cby1, _cbx2, _cby2),
                        'serial_number': state.serial_number,
                    }
                    print(f"[CROSS-CAPTURE] SEQ3 frame pre-saved while SEQ2 active "
                          f"(conf={_cross_conf:.2f} sharp={_c_sharp:.0f})")

            # SEQ3 active but SEQ2 class detected alone → rescue SEQ2 capture
            elif (state.current_sequence == 3 and _cross_name == 'panel_seq2'
                    and not state.seq2_auto_captured
                    and not state.all_sequences_done
                    and _c_sharp >= 12):
                meta = {
                    'rect':      (_cbx1, _cby1, _cw, _ch),
                    'bbox_xyxy': (_cbx1, _cby1, _cbx2, _cby2),
                    'serial_number': state.serial_number,
                }
                print(f"[CROSS-CAPTURE] SEQ2 rescued while SEQ3 active "
                      f"(conf={_cross_conf:.2f} sharp={_c_sharp:.0f})")
                capture(2, frame.copy(), metadata=meta)
                state.seq2_auto_captured = True
                state.seq_capture_data[2] = frame.copy()

    # ── SEQ2 hand-presence fallback ───────────────────────────────────────────
    # Mirrors SEQ3 fallback: when hand appears (operator starts wiping) and
    # landscape tracker has not yet fired, use the best tracked frame.
    # Fallback to last_clean when seq2_best_frame is unavailable — capture_sequence_images
    # already auto-rotates portrait frames to landscape before saving.
    if (state.current_sequence == 2
            and not state.seq2_auto_captured
            and not state.all_sequences_done
            and hand_present):
        fb = state.seq2_best_frame
        if fb is not None:
            print("[CAPTURE] SEQ2 HAND FALLBACK — using tracked horizontal frame")
            capture(2, fb, metadata=state.seq2_best_meta or {})
            state.seq2_auto_captured = True
            state.seq_capture_data[2] = fb
        else:
            print("[CAPTURE] SEQ2 HAND FALLBACK ⚠️ hand present but no landscape frame tracked — skipping"
                  " (last_clean is SEQ1 content, not suitable for SEQ2)")

    # ── SEQ2 Pi-timeout fallback ───────────────────────────────────────────────
    # Fires after 20 frames IN SEQ2 (now counted for every frame, not just wipes).
    if (state.current_sequence == 2
            and not state.seq2_auto_captured
            and not state.all_sequences_done
            and getattr(state, 'seq2_in_sequence_frames', 0) >= 20):
        fb = state.seq2_best_frame
        if fb is not None:
            print("[CAPTURE] SEQ2 TIMEOUT — using tracked horizontal frame "
                  f"(sharp={state.seq2_best_sharp:.0f})")
            capture(2, fb, metadata=state.seq2_best_meta or {})
            state.seq2_auto_captured = True
            state.seq_capture_data[2] = fb
        else:
            print("[CAPTURE] SEQ2 TIMEOUT ⚠️ 20+ frames but NO landscape frame — "
                  "will retry via complete() rescue")

    # ── SEQ3 ──────────────────────────────────────────────────────────────────
    # Same strategy as SEQ2: track best sharpness landscape frame, capture
    # after 3 confirmed horizontal frames using the sharpest one.
    seq2_also_visible = any(d['name'] == 'panel_seq2' for d in seq_cands_A)

    _seq3_det = None
    if panel_det is not None and panel_det['name'] == 'panel_seq3':
        _seq3_det = panel_det
    elif (best_any_panel is not None
          and best_any_panel['name'] == 'panel_seq3'
          # Only block when NOT in SEQ3 yet — once current_sequence==3 both panels
          # may be visible simultaneously (transition period) and we MUST capture.
          and (not seq2_also_visible or state.current_sequence == 3)):
        _seq3_det = best_any_panel

    if (state.current_sequence == 3
            and not state.seq3_auto_captured
            and not state.all_sequences_done):

        if (_seq3_det is not None
                and not hand_present
                # When current_sequence==3, allow capture even if seq2 still visible
                # (both panels in frame is normal during the transition period)
                and (not seq2_also_visible or state.current_sequence == 3)):

            bx1, by1, bx2, by2 = _seq3_det['bbox']
            _bw = max(1, bx2 - bx1)
            _bh = max(1, by2 - by1)
            _det_conf   = _seq3_det['confidence']
            _panel_area = _bw * _bh

            # PANEL-BBOX landscape check: panel must be wider than tall.
            # Fixed: was using frame dims (w>h) which always passed.
            _is_landscape = _bw > _bh * 1.1

            # Panel size gate: real panel ≥ 15% of frame.
            # Rejects empty-table false positives (bbox usually small/noisy).
            _is_big_enough = (_panel_area >= 0.15 * w * h)

            # ── Center stability: panel must not move >20px between frames
            _cx = (bx1 + bx2) // 2
            _cy = (by1 + by2) // 2
            _prev = getattr(state, 'seq3_prev_center', None)
            _is_stable = True
            if _prev is not None:
                _dx = abs(_cx - _prev[0])
                _dy = abs(_cy - _prev[1])
                _is_stable = _dx < 20 and _dy < 20
            state.seq3_prev_center = (_cx, _cy)

            if _is_landscape and _is_stable and _is_big_enough:
                frame_sharp = cv2.Laplacian(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    cv2.CV_64F).var()

                state.seq3_miss_count = 0

                if frame_sharp >= 12:
                    # FREEZE RULE: stop updating seq3_best_frame once wipe is
                    # clearly underway (>= 5 confirmed wipe frames, not just 1).
                    # 1-frame threshold was too aggressive — a single hand brush
                    # locked the frame before a good quality image was saved.
                    # 5 frames (~1s at 5fps) still protects against wipe-frame
                    # pollution while tolerating brief accidental touches.
                    _wipe_started = (state.seq3_wiping_frames >= 5)

                    # Conf gate lowered from 0.55 → 0.30.
                    # Initial panel detections are typically 0.35-0.50 (panel
                    # partly in frame, low contrast). With conf>=0.55, the very
                    # first clear landscape frames were silently rejected and
                    # seq3_best_frame stayed None → all fallbacks failed → SEQ3
                    # image was never saved.
                    if (not _wipe_started
                            and _det_conf >= 0.30
                            and frame_sharp > state.seq3_best_sharp):
                        state.seq3_best_frame = frame.copy()
                        state.seq3_best_sharp = frame_sharp
                        state.seq3_best_meta  = {
                            'rect':          (bx1, by1, _bw, _bh),
                            'bbox_xyxy':     (bx1, by1, bx2, by2),
                            'contour':       getattr(state, 'panel_contour', None),
                            'serial_number': state.serial_number,
                        }
                    state.seq3_landscape_count += 1
                    print(f"[SEQ3] Landscape #{state.seq3_landscape_count} "
                          f"sharp={frame_sharp:.0f} conf={_det_conf:.2f} "
                          f"area={_panel_area}/{int(0.15*w*h)} "
                          f"wipe_started={'Y' if _wipe_started else 'N'} "
                          f"stable={'Y' if _is_stable else 'N'}")

                    # Fire immediately on 1st frame if high-quality + pre-wipe.
                    _fire_threshold = (1 if (_det_conf >= 0.65 and frame_sharp >= 25
                                             and not _wipe_started
                                             and state.seq3_best_frame is not None)
                                       else 3)
                    if state.seq3_landscape_count >= _fire_threshold:
                        best_f = state.seq3_best_frame
                        if best_f is None and not _wipe_started:
                            # Pre-wipe — use current frame as last resort
                            best_f = frame.copy()
                        elif best_f is None and _wipe_started:
                            # Wipe in progress, no clean frame stored.
                            # frame.copy() here is a mid-wipe image — wrong.
                            # Defer to hand/timeout fallback instead.
                            print(f"[SEQ3] ⚠️ landscape≥{_fire_threshold} "
                                  f"but best_frame=None and wipe started "
                                  f"— deferring to fallback (skip mid-wipe)")
                        if best_f is not None:
                            print(f"[CAPTURE] SEQ3 ✅ confirmed "
                                  f"({state.seq3_landscape_count} frame(s), "
                                  f"threshold={_fire_threshold}, "
                                  f"sharp={state.seq3_best_sharp:.0f})")
                            capture(3, best_f, metadata=state.seq3_best_meta)
                            state.seq3_auto_captured = True
                            state.seq_capture_data[3] = best_f
                else:
                    print(f"[SEQ3] Landscape but blurry ({frame_sharp:.0f})")

            elif not _is_big_enough:
                state.seq3_miss_count = getattr(state, 'seq3_miss_count', 0) + 1
                print(f"[SEQ3] Panel too small "
                      f"({_panel_area} < {int(0.15*w*h)}) — "
                      "likely empty-table false positive, skipping")

            elif _is_landscape and not _is_stable:
                state.seq3_landscape_count = 0
                print(f"[SEQ3] MOVING — waiting for stable position")

            else:
                state.seq3_miss_count = getattr(state, 'seq3_miss_count', 0) + 1
                if state.seq3_miss_count >= 3:
                    state.seq3_landscape_count = 0
                    state.seq3_miss_count       = 0
                    print(f"[SEQ3] 3 non-landscape frames — "
                          f"resetting counter (frame {w}×{h})")
        else:
            state.seq3_miss_count = getattr(state, 'seq3_miss_count', 0) + 1
            if state.seq3_miss_count >= 5:
                state.seq3_landscape_count = 0
                state.seq3_miss_count       = 0

    # ── SEQ3 hand-presence fallback ───────────────────────────────────────────
    # STRICT: Only use tracked landscape frame — never fall back to last_clean/orig.
    if (state.current_sequence == 3
            and not state.seq3_auto_captured
            and not state.all_sequences_done
            and hand_present):
        fb = state.seq3_best_frame      # landscape frame ONLY
        if fb is not None:
            print("[CAPTURE] SEQ3 HAND FALLBACK — using tracked horizontal frame")
            capture(3, fb, metadata=state.seq3_best_meta or {})
            state.seq3_auto_captured = True
            state.seq_capture_data[3] = fb
        else:
            print("[CAPTURE] SEQ3 HAND FALLBACK ⚠️ no horizontal frame — "
                  "waiting for landscape detection")

    # ── SEQ3 Pi-timeout fallback ──────────────────────────────────────────────
    # STRICT: Only use tracked landscape frame.
    # Use seq3_landscape_count — seq3_stable_count is not incremented anywhere
    # in the codebase and would cause this fallback to NEVER fire.
    if (state.current_sequence == 3
            and not state.seq3_auto_captured
            and not state.all_sequences_done
            and getattr(state, 'seq3_landscape_count', 0) >= 15):
        fb = state.seq3_best_frame      # landscape frame ONLY
        if fb is not None:
            print("[CAPTURE] SEQ3 TIMEOUT — using tracked horizontal frame "
                  f"(sharp={state.seq3_best_sharp:.0f})")
            capture(3, fb, metadata=state.seq3_best_meta or {})
            state.seq3_auto_captured = True
            state.seq_capture_data[3] = fb
        else:
            print("[CAPTURE] SEQ3 TIMEOUT ⚠️ 15+ stable frames but NO horizontal frame — "
                  "SEQ3 image SKIPPED (client requires horizontal)")

    # ── 9. State machine ─────────────────────────────────────────
    # Reset when panel_seq1 arrives:
    #   - Case A: All 3 sequences completed → normal end-of-cycle reset.
    # ── 8. State Machine & Sequence Transitions ───────────────────
    # [RULE] Activity-Locked Transitions:
    # Only advance the sequence if we see the new class AND the operator is actually WIPING.
    # This prevents flicker from jumping to the next sequence prematurely.
    if conf_seq in (1, 2, 3):
        # Capture SEQ2 detection confidence when transitioning to SEQ2
        if conf_seq == 2 and state.current_sequence == 1 and best_any_panel:
            state.seq2_detection_confidence = best_any_panel.get('confidence', 0)
            print(f"[SEQ2] 📍 Detected with confidence {state.seq2_detection_confidence:.2f}")
        # Advance sequence immediately on detection for better UI response.
        update_seq(conf_seq)

    # Count ALL frames spent in SEQ2 — used by the timeout fallback.
    # FIX: moved here from update_wiping() where it only counted wiping frames.
    # This ensures the 20-frame timeout fires even before the operator starts wiping.
    if state.current_sequence == 2:
        state.seq2_in_sequence_frames = getattr(state, 'seq2_in_sequence_frames', 0) + 1

    # ── 8b. Wiping & Motion Detection ──────────────────────────────
    # Motion detection for wiping (Must happen AFTER update_seq so current_sequence is correct)
    gray, motion_mask = detect_motion(frame, getattr(state, 'prev_gray', None))
    state.prev_gray  = gray
    wiping_pixel     = detect_wiping(state.hand_bbox, motion_mask, state.panel_mask)
    motion           = (wiping_pixel is not None
                        and cv2.countNonZero(wiping_pixel) > 100)
    if motion and wiping_pixel is not None:
        if state.cleaned_mask is None or state.cleaned_mask.shape != wiping_pixel.shape:
            state.cleaned_mask = np.zeros_like(wiping_pixel)
        state.cleaned_mask = cv2.bitwise_or(state.cleaned_mask, wiping_pixel)
    update_wiping(hand_present, motion)

    # ── 9. State machine Resets ─────────────────────────────────────────
    # Reset when panel_seq1 arrives:
    #   - Case A: All 3 sequences completed → normal end-of-cycle reset.
    #   - Case B: We are on SEQ2 or SEQ3 and panel_seq1 re-appears → panel swap.
    if conf_seq == 1:
        if getattr(state, 'all_sequences_done', False):
            print("[PIPELINE] Cycle complete + panel_seq1 detected — resetting for new panel.")
            reset_panel()
        elif state.current_sequence in (1, 2, 3):
            # Only reset if the panel was actually physically removed for a while (> 30 frames)
            # This prevents premature resets if the model briefly hallucinates panel_seq1
            if getattr(state, 'was_absent_long', False):
                print(f"[PIPELINE] panel_seq1 seen after panel was absent during SEQ{state.current_sequence} — new panel swap, finalising old panel then resetting.")
                _finalize_panel()  # SAVE THE OLD PANEL DATA BEFORE RESET
                reset_panel()
                state.was_absent_long = False
    elif conf_seq in (2, 3):
        # The panel has stabilized to a valid non-seq1 state, so any previous absence wasn't a panel swap
        state.was_absent_long = False
    # OCR is triggered at first confirmed wiping frame on SEQ1 — see update_wiping()
    # Trigger logic moved to top of function (FIX M1)


    # ── 9b. Precise Sequence Completion & Lead Time Logic ───────────
    # [RULE] SEQ1 Completion: Only when BOTH conditions met:
    #   1. SEQ1 wiped to at least 80% coverage
    #   2. SEQ2 detected with high confidence (>= 0.50)
    #   3. Current sequence transitions to SEQ2
    if state.current_sequence == 2 and not state.completed.get(1):
        s1_pct = getattr(state, 'seq1_final_wipe_pct', 0)
        s2_conf = getattr(state, 'seq2_detection_confidence', 0)
        
        # Both gates must pass: adequate wipe on SEQ1 + strong SEQ2 detection
        _seq1_ready = (s1_pct >= 80 and s2_conf >= 0.50)
        
        if _seq1_ready:
            # Triggered by SEQ2 activity with proper wiping and confidence.
            # Audit capture (CAM2) started early at Frame 1, so it finishes 
            # in the background without stalling the state machine.
            complete(1)
            print(f"✅ [TRIGGERED] SEQ1 marked as COMPLETED | Wipe: {s1_pct:.0f}% | "
                  f"SEQ2 Confidence: {s2_conf:.2f} | Duration: {round(time.time() - state.seq_start_time.get(1, time.time()), 1)}s")
            # ── Stop OCR scanning once SEQ1 is done ──────────────────────
            # Serial number is only on the SEQ1 panel face.
            # Only stop OCR if it is already confirmed — otherwise keep collecting frames.
            if _CAM2_OCR_AVAILABLE and camera2_ocr_instance is not None:
                if camera2_ocr_instance.is_done():
                    camera2_ocr_instance._is_scanning = False
                    print('[CAM2-OCR] SEQ1 complete — OCR confirmed, scanning stopped')
                else:
                    print('[CAM2-OCR] SEQ1 complete — OCR still running; '
                          'keeping scanning alive to finish 3-slot collection')
        elif s1_pct < 80 and state.seq_start_time.get(1):
            # SEQ1 not wiped enough yet
            elapsed = time.time() - state.seq_start_time[1]
            print(f"[SEQ1] ⚠️ Panel flipped but wipe incomplete: {s1_pct:.0f}% < 80% "
                  f"(elapsed {elapsed:.1f}s) — waiting for more wiping")
        elif s2_conf < 0.50:
            # SEQ2 detected but confidence too low
            print(f"[SEQ1] ⚠️ SEQ2 detected but low confidence: {s2_conf:.2f} < 0.50 — "
                  f"requiring stronger detection")
    # ── 9c. SEQ2 Completion — TIME-BASED gates (not frame counts).
    # At 14fps:  seq2_wiping_frames>=3 = 0.2s, seq3_consec>=10 = 0.7s
    # → SEQ2 was completing in under 1 second. Frame counts are meaningless
    # at variable FPS (Pi 3fps vs 14fps).  Use elapsed wall-clock time instead.
    #
    # Requirements:
    #   • SEQ2 active for at least 3.0s  — operator genuinely wiped SEQ2
    #   • SEQ3 active for at least 2.0s  — real transition, not a detection glitch
    #   • seq2_wiping_frames >= 3        — hand actually touched SEQ2 (glitch guard)
    if state.current_sequence == 3 and not state.completed.get(2):
        # Use the SAVED wiping frame count — the live counter is reset to 0
        # during update_seq() transition.  seq2_final_wiping_frames is captured
        # at the moment of SEQ2→SEQ3 transition, before the reset.
        s2_frames  = getattr(state, 'seq2_final_wiping_frames',
                             getattr(state, 'seq2_wiping_frames', 0))
        now_t      = time.time()
        seq2_secs  = (now_t - state.seq_start_time[2]
                      if state.seq_start_time.get(2) else 0)
        seq3_secs  = (now_t - state.seq_start_time[3]
                      if state.seq_start_time.get(3) else 0)
        s2_pct     = getattr(state, 'seq2_final_wipe_pct', 0)

        # [RULE] Gated on panel flip AND SEQ2 wiped to at least 90%
        _seq2_ready = (s2_frames >= 3 and seq2_secs >= 3.0 and seq3_secs >= 2.0 and s2_pct >= 90)

        if _seq2_ready:
            # ── Wait for Camera-2 slots before closing SEQ2 ──────────────
            if (_CAM2_OCR_AVAILABLE
                    and camera2_ocr_instance is not None
                    and not camera2_ocr_instance.is_done()):
                _slots_done = all(getattr(camera2_ocr_instance,
                                          '_slots_saved', [False]*3))
                if not _slots_done:
                    # Slots still being collected — wait up to 2s
                    _t0 = time.time()
                    while time.time() - _t0 < 2.0:
                        _slots_done = all(getattr(camera2_ocr_instance,
                                                  '_slots_saved', [False]*3))
                        if _slots_done:
                            print("[SEQ2] ✅ Camera-2 slots confirmed before complete(2)")
                            break
                        time.sleep(0.1)
                    else:
                        _filled = sum(1 for s in getattr(
                            camera2_ocr_instance, '_slots_saved', []) if s)
                        print(f"[SEQ2] ⚠️ Camera-2 slots {_filled}/3 filled "
                              f"after 2s wait — proceeding with complete(2)")

            print(f"[SEQ2] ✅ Completing — seq2_frames={s2_frames} "
                  f"seq2_elapsed={seq2_secs:.1f}s seq3_elapsed={seq3_secs:.1f}s")
            if not state.seq2_auto_captured:
                print("[WARNING] SEQ2 completed, BUT image was missed!")
            complete(2)
            if state.seq_start_time.get(3):
                state.seq_end_time[2] = state.seq_start_time[3]
        elif s2_pct < 90 and seq3_secs >= 2.0:
            print(f"[SEQ2] 🚫 BLOCKED — wipe progress insufficient "
                  f"(progress={s2_pct}%, need >= 90%)")
        elif s2_frames < 3 and seq3_secs >= 2.0:
            print(f"[SEQ2] 🚫 BLOCKED — no SEQ2 wipe detected "
                  f"(frames={s2_frames}, need >= 3)")
        elif s2_frames >= 3 and seq3_secs < 2.0:
            print(f"[SEQ2] ⏳ SEQ2 touched ({s2_frames} frames, {seq2_secs:.1f}s) — "
                  f"waiting for 2.0s in SEQ3 (currently {seq3_secs:.1f}s)")
        elif s2_frames >= 3 and seq2_secs < 3.0:
            print(f"[SEQ2] ⏳ SEQ2 wiped ({s2_frames} frames) — "
                  f"waiting for 3.0s in SEQ2 (currently {seq2_secs:.1f}s)")

    # [RULE] SEQ3 Completion
    # Problem: ghost detections after removal keep resetting seq3_absence_frames.
    # Fix A — Ghost filter: after progress>=90%, only count conf>=0.65 as "present".
    # Fix B — Panel MUST fully disappear: 30+ absence frames (~1.2s at 25fps).
    # Fix C — Timeout path also requires panel to be absent (no timeout while visible).
    #
    # Rules:
    #   • 90% wipe detected (not 85%)
    #   • Panel completely gone: seq3_absence_frames >= 30
    #   • Hand removed >= 1.5s
    if state.current_sequence == 3 and not state.completed.get(3, False):

        # Ghost confidence threshold (active only after wiping threshold met)
        ghost_threshold = 0.75 if state.progress >= 90 else 0.0
        seq3_really_present = (
            conf_seq == 3
            and best_any_panel is not None
            and best_any_panel.get('confidence', 0) >= ghost_threshold
        )

        if seq3_really_present:
            state.seq3_seen_frames    += 1
            state.seq3_absence_frames  = 0
        else:
            state.seq3_absence_frames += 1

        # Calculate how long the hand has been gone (must be gone to finalize)
        hand_gone_secs = (time.time() - state.hand_gone_since) if getattr(state, 'hand_gone_since', None) else 0

        # [RULE] Normal completion:
        #   • 90%+ wipe
        #   • Panel absent 30+ consecutive frames (fully removed from table)
        #   NOTE: hand_gone_secs check removed — operator's hand stays visible
        #   while carrying the panel away, preventing completion indefinitely.
        #   seq3_absence_frames >= 30 already ensures the panel itself is gone.
        normal_done = (
            state.seq3_seen_frames >= 5
            and state.progress >= 90
            and state.seq3_absence_frames >= 30
        )

        # [RULE] Timeout completion:
        #   • 90%+ wipe, no wipe activity for 6s, panel absent
        wiping_idle_secs = (time.time() - state.last_wiping_time) \
                           if state.last_wiping_time else 99
        timeout_done = (
            state.seq3_seen_frames >= 5
            and state.progress >= 90
            and wiping_idle_secs >= 6.0
            and state.seq3_absence_frames >= 30
        )

        if normal_done or timeout_done:
            reason = "panel fully removed" if normal_done else \
                     f"wipe inactivity ({wiping_idle_secs:.1f}s) + panel gone"
            print(f"--- [SEQ3 FINALIZED] --- {reason}, "
                  f"progress={state.progress}% "
                  f"absence={state.seq3_absence_frames} frames")
            if not state.seq3_auto_captured:
                print("--- [WARNING] --- SEQ3 image was missed!")
            complete(3)


    # ── Camera-2 OCR result check (ADD ONLY — no Camera-1 change) ───
    # Poll the OCR thread non-blockingly; update serial when ready.
    # ── Camera-2 OCR result check ─────────────────────────────────────
    if state.ocr_started and _CAM2_OCR_AVAILABLE and camera2_ocr_instance:
        cam2_serial  = camera2_ocr_instance.get_serial_number()
        cam2_partial = getattr(camera2_ocr_instance, 'partial_serial', None)
        # Update partial for live feedback (exclude non-meaningful values)
        if cam2_partial and cam2_partial not in ('Reading...', 'UNKNOWN'):
            state.partial_serial = cam2_partial
        # Update confirmed serial (only real values — not placeholders)
        # FIX: was "if not state.ocr_done" which fired once when serial='Reading...'
        # (Phase-1 just captured the frame) then BLOCKED the real serial update
        # when background OCR finished later.  Now we gate on the CURRENT
        # state.serial_number value instead — update whenever we get a real serial.
        if (camera2_ocr_instance.is_done()
                and cam2_serial
                and cam2_serial not in ('UNKNOWN', 'Reading...')):
            if state.serial_number in (None, 'UNKNOWN', 'Searching...', 'Reading...', ''):
                state.serial_number = cam2_serial
                state.ocr_done      = True
                print(f'[CAM2-OCR] Serial confirmed and pushed to UI: {cam2_serial}')

    if state.ocr_done and not getattr(state, 'folder_renamed', False):
        try:
            old_path = state.current_sequence_panel_folder
            serial   = state.serial_number

            if (old_path and os.path.exists(old_path)
                    and serial and serial not in ('UNKNOWN', 'Reading...')):
                # Only rename if serial differs from current folder name
                if os.path.basename(old_path) != serial:
                    new_path = os.path.join(os.path.dirname(old_path), serial)
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        state.current_sequence_panel_folder = new_path
                        print(f"[FOLDER] Renamed → {new_path}")
                    else:
                        state.current_sequence_panel_folder = new_path

                    # BUG FIX: camera2_ocr_instance still holds the OLD folder
                    # path in .panel_folder, .cam2_raw_path, .cam2_roi_path.
                    # _background_ocr uses these — if stale it bails immediately
                    # (cv2.imread on old path → None) and ALL cam2_temp debug
                    # files + file renames never happen.
                    # Update them here so background_ocr resolves the new path.
                    if _CAM2_OCR_AVAILABLE and camera2_ocr_instance:
                        camera2_ocr_instance.panel_folder = new_path
                        for attr in ('cam2_raw_path', 'cam2_roi_path'):
                            p = getattr(camera2_ocr_instance, attr, None)
                            if p and old_path in p:
                                setattr(camera2_ocr_instance, attr,
                                        p.replace(old_path, new_path, 1))
                                print(f"[FOLDER] cam2 path updated: {attr}")

                # FIX: rename image files too — was missing, so files kept
                # "Searching..." prefix even after the folder was renamed.
                rename_panel_images(state.current_sequence_panel_folder, serial)

                state.folder_renamed = True

                # Save best Camera-2 frame to panel folder as Serial_Original.
                # Priority: cam2_raw_path (best slot frame already saved to disk)
                # → annotated frame of that moment as fallback.
                try:
                    dest       = state.current_sequence_panel_folder
                    serial_orig = os.path.join(dest,
                                               f"{serial}_Serial_Original.jpg")
                    # Use the best slot frame path if it exists on disk
                    best_path = getattr(camera2_ocr_instance,
                                        'cam2_raw_path', None)
                    if best_path and os.path.exists(best_path):
                        import shutil
                        shutil.copy2(best_path, serial_orig)
                        state.cam2_image_path = serial_orig
                        print(f"[CAM2] Serial_Original copied from best slot → "
                              f"{os.path.basename(serial_orig)}")
                    else:
                        # Fallback: latest Camera-2 frame
                        cam2_img = (camera2_ocr_instance.get_annotated_frame()
                                    if camera2_ocr_instance else None)
                        if cam2_img is not None:
                            cv2.imwrite(serial_orig, cam2_img)
                            state.cam2_image_path = serial_orig
                            print(f"[CAM2] Serial_Original saved (latest frame) → "
                                  f"{os.path.basename(serial_orig)}")
                except Exception as _ce:
                    print(f"[CAM2] Serial_Original save error: {_ce}")
        except Exception as _re:
            print(f"[FOLDER] Rename error: {_re}")

    if not getattr(state, 'cam2_frame_saved', False):
        try:
            # FIX A3: gate on active panel session — prevents premature folder
            # creation between panels when cam2_frame_saved is False after reset.
            if (_CAM2_OCR_AVAILABLE and camera2_ocr_instance
                    and state.panel_id is not None
                    and state.current_sequence > 0):
                # ── Priority 1: Tight ROI (Zoomed preferred)
                p = getattr(camera2_ocr_instance, 'cam2_roi_path', None)
                if not p:
                    p = getattr(camera2_ocr_instance, 'cam2_raw_path', None)
                
                # ── Priority 2: In-progress or failed frame (Attempt_ prefix)
                if not p or not os.path.exists(p):
                    folder = ensure_panel_folder()
                    import glob
                    # Prioritize Attempt ROI over Attempt Full
                    f = glob.glob(os.path.join(folder, "Attempt_*_Cam2_ROI_*.jpg"))
                    if not f:
                        f = glob.glob(os.path.join(folder, "Attempt_Cam2_OCR_Reference_Full_*.jpg"))
                    if f: p = f[0]
                
                if p and os.path.exists(p):
                    state.cam2_image_path = p
                    state.cam2_frame_saved = True
                    print(f"[CAM2] Linked tight ROI frame to PDF: {os.path.basename(p)}")
        except Exception as e:
            print(f"[CAM2 LINK ERROR] {e}")

    # ── 10. Debug log every 30 frames ──────────────────────────────
    state.frame_count = getattr(state, 'frame_count', 0) + 1
    if state.frame_count % 30 == 0:
        print(f"  [STATE] SEQ={state.current_sequence} "
              f"prog={state.progress}% "
              f"wipe={state.total_wipe_seconds:.1f}s "
              f"model_sees={getattr(state,'model_sees','-')} "
              f"conf_seq={conf_seq}")

    # FIX I2: model_sees uses TRACK A (what model actually outputs).
    # TRACK B (panel_det) is class-locked so it returns None during
    # misclassification — that was causing model_sees='NONE' and blank UI class.
    state.model_sees = best_any_panel['name'] if best_any_panel else "NONE"

    # ── 11. Update held annotation state (stable, no-flicker overlay) ────────
    # update_held_boxes keeps confirmed bboxes alive for PANEL_HOLD_FRAMES so
    # draw_overlay never flickers even when inference is slower than the stream.
    update_held_boxes(detections, conf_seq)
    # Keep a copy for OCR ROI-crop logic in run_ocr() — NOT used for display.
    state.latest_detections = detections

    # ── 12. Draw overlay + blue tint (on a copy for the UI, original stays clean)
    out = apply_blue_tint(frame.copy(), state.panel_mask, state.cleaned_mask)
    out = draw_overlay(out, state.current_sequence, state.status_msg, state.progress)
    return out



# ─────────────────────────────────────────────────────────────
# CAMERA STREAMING
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# VIDEO SOURCE STATE
# ─────────────────────────────────────────────────────────────
# Holds the path to an uploaded video file (None = live camera)
state.video_source = None   # set by /api/upload_video


# ─────────────────────────────────────────────────────────────
# FRAME GENERATORS
# ─────────────────────────────────────────────────────────────

def _stream(cap, native_fps=None):
    """
    4-thread pipeline:
      READER  → puts raw frames into raw_q (always latest)
      ENCODER → reads raw_q, encodes JPEG, puts into stream_q (always latest)
      INFER   → reads raw_q independently, runs inference, updates overlay
      STREAMER → yields from stream_q at full speed

    Stream is NEVER blocked by inference. Inference runs independently
    and updates a shared overlay that ENCODER applies to raw frames.
    """
    import queue as _queue

    my_generation = state.stream_generation
    running       = threading.Event()
    running.set()

    # Shared: latest raw frame and latest overlay
    raw_lock      = threading.Lock()
    raw_frame     = [None]     # latest BGR frame
    raw_frame_cnt = [0]        # monotonic counter — increments every new frame
    overlay_lock  = threading.Lock()
    overlay_frame = [None]     # latest inference result (annotated frame)

    # Output queue: encoder → streamer
    stream_q = _queue.Queue(maxsize=2)

    # ── Thread 1: READER ─────────────────────────────────────────
    def reader():
        nonlocal cap
        source_url  = state.video_source
        is_rtsp     = isinstance(source_url, str) and source_url.startswith('rtsp://')
        is_file     = source_url and os.path.exists(str(source_url)) and not is_rtsp
        is_tls      = getattr(cap, '__class__', None) and cap.__class__.__name__ == '_TLSRTSPCapture'
        is_ffmpeg   = getattr(cap, '__class__', None) and cap.__class__.__name__ == 'FFmpegVideoCapture'
        
        # FIX C1: On Pi, TLS-RTSP negotiation takes 10-30 s before isOpened().
        # Old code returned permanently on False, killing the reader thread.
        # Now we spin-wait up to 60 s then fall through to the reconnect loop.
        _open_wait = 0
        while not cap.isOpened() and _open_wait < 60:
            time.sleep(1.0)
            _open_wait += 1
            if _open_wait % 10 == 0:
                print(f"[READER] ⏳ Waiting for camera to open ... {_open_wait}s")
        if not cap.isOpened():
            print("[READER] Camera not open after 60 s - entering reconnect loop")
            with raw_lock: raw_frame[0] = None
            # Do NOT return - fall through to main while loop for reconnect

        # Safety cap for playback speed
        target_fps = min(native_fps, 30.0) if native_fps else 15.0
        print(f"[READER] ✅ Thread started for {source_url} (Target FPS: {target_fps})")
        
        # Ensure we don't start with a None frame
        ok, first_frame = cap.read()
        if ok:
            with raw_lock:
                raw_frame[0] = first_frame.copy()
                raw_frame_cnt[0] += 1
        
        fails = 0
        last_ok = time.time()
        last_read_time = time.time()
        # FIX C2: 8 s -> 15 s - TLS-RTSP reconnect on Pi takes up to 12 s;
        # 8 s caused a reconnect storm that saturated the Pi CPU.
        STALE_TIMEOUT = 15.0

        while running.is_set() and state.stream_generation == my_generation:
            # ── Playback Throttling (for video files) ──────────────────
            if is_file:
                interval = 1.0 / target_fps
                elapsed = time.time() - last_read_time
                if elapsed < interval:
                    time.sleep(interval - elapsed)
            
            last_read_time = time.time()

            ok, frame = cap.read()
            if not ok:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    fails = 0
                    continue
                
                fails += 1
                if fails % 30 == 0:
                    print(f"[READER] ⚠️ {fails} consecutive read failures from {source_url}")
                time.sleep(0.05)
                # [FIX] Hold the last frame for 5 seconds (fails==100) before going completely black.
                if fails == 100:
                    with raw_lock: raw_frame[0] = None
                # ── Stale-frame watchdog: force reconnect if silent drop ──
                if time.time() - last_ok > STALE_TIMEOUT and not is_file:
                    print(f'[READER] ⚠️ Stream silent for {STALE_TIMEOUT}s — forcing reconnect')
                    try: cap.release()
                    except: pass
                    time.sleep(1)
                    cap = _open_rtsp(source_url)
                    is_tls = getattr(cap, '__class__', None) and cap.__class__.__name__ == '_TLSRTSPCapture'
                    fails = 0
                    last_ok = time.time()
                    # Reset prev_gray — new stream may have different resolution
                    state.prev_gray = None
                elif fails >= (600 if is_tls else 30):  # 600 * 0.05s = 30s to match timeouts
                    if is_ffmpeg:
                        fails = 0
                    elif is_file:
                        print(f"[READER] ❌ Permanent failure reading file {source_url}")
                        running.clear()
                        break
                    else:
                        print(f'[READER] ⚠️ reconnecting RTSP after {time.time()-last_ok:.1f}s')
                        try:
                            cap.release()
                            time.sleep(0.5)
                        except: pass
                        cap = _open_rtsp(source_url)
                        is_tls = getattr(cap, '__class__', None) and cap.__class__.__name__ == '_TLSRTSPCapture'
                        fails = 0
                        # Reset prev_gray — new stream may have different resolution
                        state.prev_gray = None
                continue
            
            fails = 0; last_ok = time.time()
            with raw_lock:
                raw_frame[0]      = frame.copy()
                raw_frame_cnt[0] += 1

        try: cap.release()
        except: pass

    # ── Thread 2: ENCODER (stream thread — never blocked by inference) ──
    def encoder():
        last_enc_time  = time.time()
        last_frame_ts  = time.time()   # track when last real frame arrived
        STALE_BANNER   = 4.0           # show reconnecting banner after 4s no frame
        MIN_INTERVAL   = 1.0 / STREAM_FPS_CAP
        STREAM_W       = STREAM_UI_WIDTH

        def _reconnecting_banner():
            """Return a JPEG banner shown when Camera-1 has no live frame."""
            blank = np.zeros((960, STREAM_W, 3), dtype=np.uint8)
            cv2.rectangle(blank, (0, 0), (STREAM_W, 960), (0, 80, 200), 8)
            cv2.putText(blank, 'CAMERA 1 RECONNECTING...',
                        (60, 480), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 80, 200), 3)
            _, b = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
            return b.tobytes()

        while running.is_set() and state.stream_generation == my_generation:
            elapsed = time.time() - last_enc_time
            if elapsed < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - elapsed)
                continue

            with raw_lock:
                frame = raw_frame[0]

            if frame is None:
                # No frame at all — check how long we've been waiting
                if time.time() - last_frame_ts > STALE_BANNER:
                    # Push reconnecting banner so browser doesn't freeze
                    try:
                        stream_q.get_nowait()
                    except _queue.Empty:
                        pass
                    stream_q.put_nowait(_reconnecting_banner())
                    last_enc_time = time.time()
                time.sleep(0.1)
                continue

            last_frame_ts = time.time()    # real frame arrived — reset
            h, w = frame.shape[:2]
            sh   = int(h * (STREAM_W / w))

            try:
                disp  = apply_blue_tint(frame.copy(), state.panel_mask, state.cleaned_mask)
                disp  = draw_overlay(disp, state.current_sequence,
                                     getattr(state, 'status_msg', ''),
                                     getattr(state, 'progress', 0))
                small = cv2.resize(disp, (STREAM_W, sh),
                                   interpolation=cv2.INTER_LINEAR)
                _, buf = cv2.imencode('.jpg', small,
                                      [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                jpg = buf.tobytes()

                try: stream_q.get_nowait()
                except _queue.Empty: pass
                stream_q.put_nowait(jpg)
                last_enc_time = time.time()
            except Exception as e:
                print(f"[ENCODER ERR] {e}")

        stream_q.put(None)

    # ── Thread 3: INFERENCE (throttled to ~5 fps — never blocks stream) ──
    # CPU YOLO takes 200-400ms/frame. Run-then-sleep keeps encoder free.
    INFER_INTERVAL = 0.20   # 5 fps

    def infer():
        last_cnt = [-1]
        print('[INFER] Thread started')

        while running.is_set() and state.stream_generation == my_generation:

            with raw_lock:
                frame = raw_frame[0]
                cnt   = raw_frame_cnt[0]

            if frame is None:
                time.sleep(0.05)
                continue

            # Skip if no new frame since last run
            if cnt == last_cnt[0]:
                time.sleep(0.05)
                continue
            last_cnt[0] = cnt

            try:
                h, w = frame.shape[:2]
                # Store full-res original for PDF capture
                state.orig_frame = frame.copy()

                # 1. Resize for AI model; rescale bboxes back to full-res
                if w > MAX_PROC_WIDTH:
                    sx = MAX_PROC_WIDTH / w
                    small = cv2.resize(frame, (0, 0), fx=sx, fy=sx,
                                       interpolation=cv2.INTER_AREA)
                    detections = detect_hand(small)
                    for d in detections:
                        x1, y1, x2, y2 = d['bbox']
                        d['bbox'] = (int(x1/sx), int(y1/sx),
                                     int(x2/sx), int(y2/sx))
                        if d.get('polygon') is not None:
                            d['polygon'] = (d['polygon'] / sx).astype(np.int32)
                    # Scale factors: full_dim / small_dim
                    state._scale_x = 1.0 / sx
                    state._scale_y = 1.0 / sx
                else:
                    detections = detect_hand(frame)
                    state._scale_x = 1.0
                    state._scale_y = 1.0

                # 2. Run the full processing pipeline
                result = process_frame(frame, detections)

                # ── [DEBUG 6] Pipeline Flow Check ─────────────────────────
                if state.frame_count % 30 == 0:
                    print(f"[PIPELINE] detections passed to overlay layer: {len(detections)}")

                # 3. Send annotated result to encoder
                with overlay_lock:
                    overlay_frame[0] = result

                # ── FIX O1: Periodic Garbage Collection ────────────────────
                if state.frame_count % 100 == 0:
                    gc.collect()

            except Exception as e:
                import traceback
                print(f"[INFER THREAD ERROR] {e}")
                traceback.print_exc()
                time.sleep(0.1)

            # Sleep AFTER inference so the encoder always gets CPU time
            time.sleep(INFER_INTERVAL)

    # Start threads
    for t in [threading.Thread(target=reader,  daemon=True),
              threading.Thread(target=encoder, daemon=True),
              threading.Thread(target=infer,   daemon=True)]:
        t.start()

    # ── STREAMER (generator) ──────────────────────────────────────
    while True:
        try:
            buf = stream_q.get(timeout=0.5)
        except _queue.Empty:
            if not state.camera_active or \
               state.stream_generation != my_generation:
                break
            continue

        if buf is None:
            break

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buf + b'\r\n')

    running.clear()


def _start_tls_proxy(host: str, cam_port: int, local_port: int) -> bool:
    """
    Start a local TCP proxy that wraps TLS -> plain TCP.
    Each camera gets its own dedicated local_port.
    """
    import socket as _socket, ssl as _ssl, threading as _threading
    
    # Track which ports are already handled to avoid restarting
    if not hasattr(_start_tls_proxy, '_active_ports'):
        _start_tls_proxy._active_ports = set()
    
    if local_port in _start_tls_proxy._active_ports:
        return True

    def handle(client):
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode    = _ssl.CERT_NONE
        try:
            raw = _socket.create_connection((host, cam_port), timeout=10)
            # Enable TCP_NODELAY and increase buffers
            raw.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
            try:
                raw.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, 1048576)
                raw.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, 1048576)
            except: pass
            cam = ctx.wrap_socket(raw, server_hostname=host)
        except Exception as e:
            print(f'[proxy] cam connect failed: {e}')
            client.close(); return

        def pipe(src, dst):
            try:
                while True:
                    d = src.recv(8192)
                    if not d: break
                    dst.sendall(d)
            except Exception: pass
            try: src.close()
            except: pass
            try: dst.close()
            except: pass

        _threading.Thread(target=pipe, args=(client, cam), daemon=True).start()
        _threading.Thread(target=pipe, args=(cam, client), daemon=True).start()

    def server():
        try:
            srv = _socket.socket()
            srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            srv.bind(('127.0.0.1', local_port))
            # Increase server buffers
            try:
                srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, 1048576)
                srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_SNDBUF, 1048576)
            except: pass
            srv.listen(10)
            print(f'[proxy] ✅ Proxy active: 127.0.0.1:{local_port} -> {host}:{cam_port}')
            while True:
                try:
                    client, _ = srv.accept()
                    client.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
                    _threading.Thread(target=handle, args=(client,), daemon=True).start()
                except Exception as e:
                    print(f'[proxy] Accept error on {local_port}: {e}')
                    break
        except Exception as e:
            print(f'[proxy] ❌ Could not start proxy on {local_port}: {e}')

    _threading.Thread(target=server, daemon=True).start()
    _start_tls_proxy._active_ports.add(local_port)
    time.sleep(1.0) # Give it a full second to stabilize
    return True


# ─────────────────────────────────────────────────────────────────────
# HWACCEL DETECTION — run once at startup, shared by all FFmpeg cmds
# ─────────────────────────────────────────────────────────────────────

def _detect_ffmpeg_hwaccel():
    """
    Returns 'nvidia' | 'v4l2m2m' | 'software'.
    Called once at startup; result cached in _HW_TYPE.
    """
    try:
        r   = subprocess.run(['ffmpeg', '-hwaccels'],
                             capture_output=True, text=True, timeout=4)
        out = r.stdout.lower()
        is_arm   = ('arm'   in platform.machine().lower() or
                    'aarch' in platform.machine().lower())
        has_cuda  = 'cuda'     in out or 'cuvid' in out
        has_v4l2  = 'v4l2m2m' in out

        if has_cuda and not is_arm:
            print('[HWACCEL] NVIDIA GPU detected → h264_cuvid ✅')
            return 'nvidia'
        if is_arm and has_v4l2:
            print('[HWACCEL] ARM Pi → software decode (v4l2m2m disabled)')
            return 'software'
    except Exception as e:
        print(f'[HWACCEL] probe failed ({e})')
    print('[HWACCEL] Software decode (libavcodec)')
    return 'software'

_HW_TYPE, _HW_V4L2 = _detect_ffmpeg_hwaccel(), False


def _hw_prefix():
    """
    Returns FFmpeg flags to insert BEFORE -i.
    NVIDIA: h264_cuvid GPU decode, output in system RAM (vf works fine).
    Other:  software decode.
    """
    if _HW_TYPE == 'nvidia':
        return ['-hwaccel', 'cuvid', '-c:v', 'h264_cuvid']
    return []


def _open_rtsp(url: str, width=1280, height=960):
    if ':443/' in url or ':443' in url:
        return _TLSRTSPCapture(url, width=width, height=height)
    return FFmpegVideoCapture(url, width=width, height=height)


class _TLSRTSPCapture:
    def isOpened(self):
        return getattr(self, '_opened', False)
    def release(self):
        self._running = False
        if self._proc:
            try: self._proc.kill()
            except: pass

    """
    RTSP over TLS with Digest auth — for CP Plus cameras (port 443).
    Also tries plain ffmpeg as fallback for other cameras.
    Mimics cv2.VideoCapture interface.
    """
    def __init__(self, url: str, width=1280, height=960):
        self._url      = url
        self._opened   = False
        self._w        = width
        self._h        = height
        self._proc     = None
        self._tls_sock = None
        self._lock     = threading.Lock()
        self._frame_q  = []
        self._thread   = None
        self._running  = False
        self._parse_url(url)
        self._start()

    def _parse_url(self, url: str):
        """Parse rtsp://user:pass@host:port/path"""
        import re
        m = re.match(
            r'rtsp://([^:@]+):([^@]*)@([^:/]+):?(\d*)(.*)', url)
        if m:
            self._user = m.group(1)
            self._pwd  = m.group(2)
            self._host = m.group(3)
            self._port = int(m.group(4)) if m.group(4) else 554
            self._path = m.group(5) or '/video/live?channel=1&subtype=0'
        else:
            m2 = re.match(r'rtsp://([^:/]+):?(\d*)(.*)', url)
            self._user = ''
            self._pwd  = ''
            self._host = m2.group(1) if m2 else '127.0.0.1'
            self._port = int(m2.group(2)) if m2 and m2.group(2) else 554
            self._path = m2.group(3) if m2 else '/video/live?channel=1&subtype=0'

    def _start(self):
        """Try direct ffmpeg rtsps first, then TLS socket fallback."""
        self._running = True   # must be True before reader threads start
        
        # 1. First try native ffmpeg with rtsps://
        if self._try_native_rtsps():
            return
            
        # 2. Try proxy ffmpeg
        if self._try_ffmpeg_direct():
            return
            
        # 3. Try python tls socket
        if self._try_tls_rtsp():
            return
            
        print("[RTSP] ❌ All methods failed")

    def _try_native_rtsps(self) -> bool:
        """Native rtsps:// in ffmpeg (skipping proxy)."""
        rtsps_url = self._url.replace("rtsp://", "rtsps://")
        
        self._dec_w = 1280
        self._dec_h = 960
        frame_size  = self._dec_w * self._dec_h * 3

        # Pure software H264 decode — correct BGR24 output.
        # probesize/analyzeduration NOT restricted: FFmpeg needs enough bytes
        # to read H264 SPS/PPS headers or the decoder can't initialise.
        cmd = ['ffmpeg', '-y', '-loglevel', 'warning',
               '-user_agent', 'LibVLC',
               '-tls_verify', '0',
               '-rtsp_transport', 'tcp',
               '-rtsp_flags',     'prefer_tcp',
               '-fflags',         'nobuffer+discardcorrupt',
               '-flags',          'low_delay',
               '-i',              rtsps_url,
               # Limit output to 15 fps — reduces pipe backlog without losing freshness
               '-r',              '20',
               '-vf',             f'scale={self._dec_w}:{self._dec_h},format=bgr24',
               '-vcodec',         'rawvideo',
               '-f',              'rawvideo',
               '-vsync',          '0',
               '-an', '-sn', 'pipe:1']
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # REDUCED: was 4 MB — caused multiple frames to buffer
                # and produced 5-20 s of visual lag.
                bufsize=frame_size)

            # REDUCED timeout: 12 s (was 30 s)
            r, _, _ = select.select([proc.stdout], [], [], 12)
            if r:
                raw = proc.stdout.read(frame_size)
                if len(raw) == frame_size:
                    self._proc   = proc
                    self._opened = True
                    frame = np.frombuffer(raw, dtype=np.uint8
                            ).reshape((self._dec_h, self._dec_w, 3)).copy()
                    with self._lock:
                        self._frame_q = [frame]
                    print(f'[RTSP] ✅ native rtsps {self._dec_w}x{self._dec_h}')
                    
                    def consume_stderr(p):
                        while self._running and p.poll() is None:
                            try:
                                line = p.stderr.readline()
                                if not line: break
                            except: break
                    threading.Thread(target=consume_stderr, args=(proc,), daemon=True).start()
                    threading.Thread(target=self._ffmpeg_frame_reader, daemon=True).start()
                    return True
            proc.kill()
            err = proc.stderr.read(500).decode('utf-8', errors='ignore')
            print(f'[RTSP] native rtsps failed: {err[-150:]}')
        except Exception as e:
            print(f'[RTSP] native rtsps error: {e}')
        return False


    def _try_ffmpeg_direct(self) -> bool:
        """Use local TLS proxy + ffmpeg."""
        import socket as _sock, select as _sel
        
        # [FIX] Assign unique proxy ports per camera IP to avoid collisions
        # Camera 1 (188) -> 5554, Camera 2 (192) -> 5555
        proxy_port = 5554 if '.188' in self._host else 5555
        _start_tls_proxy(self._host, 443, proxy_port)
        
        # Wait until proxy is actually listening
        for _ in range(30):
            try:
                t = _sock.create_connection(('127.0.0.1', proxy_port), timeout=0.1)
                t.close(); break
            except Exception:
                time.sleep(0.1)

        proxy_url = (f'rtsp://{self._user}:{self._pwd}@'
                     f'127.0.0.1:{proxy_port}{self._path}')

        # Force 1280x960 for Camera 1
        self._dec_w = 1280
        self._dec_h = 960
        frame_size  = self._dec_w * self._dec_h * 3

        # Pure software H264 decode — correct BGR24 output.
        cmd = ['ffmpeg', '-y', '-loglevel', 'warning',
               '-rtsp_transport', 'tcp',
               '-fflags',         'nobuffer+discardcorrupt',
               '-flags',          'low_delay',
               '-i',              proxy_url,
               '-r',              '20',
               '-vf',             f'scale={self._dec_w}:{self._dec_h},format=bgr24',
               '-vcodec',         'rawvideo',
               '-f',              'rawvideo',
               '-vsync',          '0',
               '-an', '-sn', 'pipe:1']
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # REDUCED: was 4 MB buffer
                bufsize=frame_size)

            # REDUCED timeout: 12 s (was 30 s)
            r, _, _ = select.select([proc.stdout], [], [], 12)
            if r:
                raw = proc.stdout.read(frame_size)
                if len(raw) == frame_size:
                    self._proc   = proc
                    self._opened = True
                    frame = np.frombuffer(raw, dtype=np.uint8
                            ).reshape((self._dec_h, self._dec_w, 3)).copy()
                    with self._lock:
                        self._frame_q = [frame]
                    print(f'[RTSP] ✅ ffmpeg proxy {self._dec_w}x{self._dec_h} ~14fps')
                    
                    def consume_stderr(p):
                        while self._running and p.poll() is None:
                            try:
                                line = p.stderr.readline()
                                if not line: break
                            except: break
                    threading.Thread(target=consume_stderr, args=(proc,), daemon=True).start()
                    threading.Thread(target=self._ffmpeg_frame_reader, daemon=True).start()
                    return True
            proc.kill()
            err = proc.stderr.read(500).decode('utf-8', errors='ignore')
            print(f'[RTSP] ffmpeg proxy failed: {err[-150:]}')
        except Exception as e:
            print(f'[RTSP] ffmpeg proxy error: {e}')
        return False

    def _ffmpeg_frame_reader(self):
        """Read BGR frames from ffmpeg — always discard old, keep only latest."""
        import select
        frame_size = self._dec_w * self._dec_h * 3
        _buf = bytearray(frame_size)
        _mv  = memoryview(_buf)

        while self._running and self._proc:
            try:
                readable, _, _ = select.select([self._proc.stdout], [], [], 8.0)
                if not readable:
                    print('[RTSP] ffmpeg reader timed out (8s) — reconnecting')
                    break

                # Read exactly one full frame — sequential read, never partial.
                # Do NOT use non-blocking drain: partial reads desync frame
                # boundaries → pixels from two frames mix → blur/corruption.
                n = 0
                try:
                    while n < frame_size:
                        got = self._proc.stdout.readinto(_mv[n:])
                        if not got:
                            break
                        n += got
                except Exception:
                    break

                if n != frame_size:
                    print('[RTSP] Incomplete frame — skipping')
                    continue

                frame = np.frombuffer(_buf, dtype=np.uint8
                        ).reshape((self._dec_h, self._dec_w, 3)).copy()
                with self._lock:
                    self._frame_q = [frame]   # always replace — never queue
                    self._last_frame_ts = time.time()

            except Exception:
                break
        print('[RTSP] frame reader stopped')

    def _make_tls_socket(self, port):
        """Create TLS socket to camera."""
        import ssl, socket as _socket
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        raw = _socket.create_connection((self._host, port), timeout=8)
        return ctx.wrap_socket(raw, server_hostname=self._host)

    def _digest_auth(self, method, url, realm, nonce):
        import hashlib
        def md5(s): return hashlib.md5(s.encode()).hexdigest()
        ha1 = md5(f'{self._user}:{realm}:{self._pwd}')
        ha2 = md5(f'{method}:{url}')
        resp = md5(f'{ha1}:{nonce}:{ha2}')
        return (f'Digest username="{self._user}", realm="{realm}", '
                f'nonce="{nonce}", uri="{url}", response="{resp}"')

    def _try_tls_rtsp(self) -> bool:
        """RTSP-over-TLS on port 443 with Digest auth. Proven working."""
        import re, time, base64
        rtsp_url = 'rtsp://' + self._host + ':554' + self._path
        try:
            s = self._make_tls_socket(443)

            # OPTIONS -> get realm/nonce
            s.send(('OPTIONS ' + rtsp_url + ' RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: LibVLC\r\n\r\n').encode())
            time.sleep(0.3)
            r1 = s.recv(4096).decode('utf-8', errors='ignore')
            rm = re.search(r'realm="([^"]+)"', r1)
            nm = re.search(r'nonce="([^"]+)"', r1)
            if not rm: s.close(); return False
            realm = rm.group(1); nonce = nm.group(1)

            # DESCRIBE -> SDP + SPS/PPS
            auth = self._digest_auth('DESCRIBE', rtsp_url, realm, nonce)
            s.send(('DESCRIBE ' + rtsp_url + ' RTSP/1.0\r\nCSeq: 2\r\nUser-Agent: LibVLC\r\n' +
                    'Authorization:' + auth + '\r\nAccept: application/sdp\r\n\r\n').encode())
            time.sleep(0.3)
            r2 = s.recv(8192).decode('utf-8', errors='ignore')
            if '200 OK' not in r2: s.close(); return False

            spspps = b''
            fmtp = re.search(r'sprop-parameter-sets=([^;\s\r\n]+)', r2)
            if fmtp:
                for nb64 in fmtp.group(1).split(','):
                    try: spspps += b'\x00\x00\x00\x01' + base64.b64decode(nb64 + '==')
                    except: pass
            self._spspps = spspps
            print('[RTSP] ✅ TLS DESCRIBE 200 OK — SPS/PPS ' + str(len(spspps)) + ' bytes')

            # SETUP
            track_m   = re.search(r'a=control:(trackID=\d+)', r2)
            track     = track_m.group(1) if track_m else 'trackID=0'
            setup_url = rtsp_url + '/' + track
            auth3 = self._digest_auth('SETUP', setup_url, realm, nonce)
            s.send(('SETUP ' + setup_url + ' RTSP/1.0\r\nCSeq: 3\r\nUser-Agent: LibVLC\r\n' +
                    'Authorization:' + auth3 + '\r\n' +
                    'Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n').encode())
            time.sleep(0.3)
            r3 = s.recv(4096).decode('utf-8', errors='ignore')
            sm = re.search(r'Session:\s*([^\r\n;]+)', r3)
            session = sm.group(1).strip() if sm else ''
            print('[RTSP] SETUP: ' + r3.split('\n')[0].strip())

            # PLAY
            auth4 = self._digest_auth('PLAY', rtsp_url, realm, nonce)
            s.send(('PLAY ' + rtsp_url + ' RTSP/1.0\r\nCSeq: 4\r\nUser-Agent: LibVLC\r\n' +
                    'Authorization:' + auth4 + '\r\n' +
                    'Session: ' + session + '\r\nRange: npt=0.000-\r\n\r\n').encode())
            time.sleep(0.3)
            r4 = s.recv(4096).decode('utf-8', errors='ignore')
            if '200 OK' not in r4: s.close(); return False

            print('[RTSP] ✅ PLAY started — reading RTP over TLS socket')
            self._tls_sock = s
            self._opened   = True
            self._proc     = None
            self._running  = True
            self._thread   = threading.Thread(target=self._rtp_reader, daemon=True)
            self._thread.start()

            for _ in range(100):
                time.sleep(0.1)
                with self._lock:
                    if self._frame_q:
                        print('[RTSP] ✅ First frame received!')
                        return True
            print('[RTSP] ⚠️  PLAY ok but no frames yet — continuing')
            return True
        except Exception as e:
            print('[RTSP] TLS attempt failed: ' + str(e))
            return False

    def _rtp_reader(self):
        """RTP over TLS → H264 annexb → ffmpeg → BGR frames at real-time speed."""
        import struct, queue as _q

        spspps  = getattr(self, '_spspps', b'')
        dec_w   = 1280
        dec_h   = 960   # FIXED: was 720 — must match native camera resolution

        proc = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'warning',
             # probesize NOT restricted (was 32) — must read SPS/PPS headers
             '-flags', 'low_delay',
             '-fflags', 'nobuffer+discardcorrupt',
             '-f', 'h264', '-i', 'pipe:0',
             '-vf', f'scale={dec_w}:{dec_h},format=bgr24',
             '-f', 'rawvideo', '-pix_fmt', 'bgr24',
             '-vsync', 'drop',
             'pipe:1'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Stderr reader thread for debugging
        def stderr_reader():
            while self._running:
                try:
                    line = proc.stderr.readline()
                    if not line: break
                    print(f'[RTSP-FFMPEG] {line.decode("utf-8", errors="ignore").strip()}')
                except: break
        threading.Thread(target=stderr_reader, daemon=True).start()

        frame_size = dec_w * dec_h * 3

        # Frame reader thread — continuously reads decoded BGR frames
        def frame_reader():
            import select as _sel
            last_data = [time.time()]
            WATCHDOG  = 12.0          # restart if no frame for 12s (give time for IDR)
            while self._running:
                try:
                    # select() with 1s timeout — prevents blocking forever
                    rdy, _, _ = _sel.select([proc.stdout], [], [], 1.0)
                    if not rdy:
                        if time.time() - last_data[0] > WATCHDOG:
                            print(f'[RTSP] frame_reader: watchdog timeout ({WATCHDOG}s) — restarting')
                            proc.kill()
                            break
                        continue
                    raw = proc.stdout.read(frame_size)
                    if len(raw) == frame_size:
                        last_data[0] = time.time()  # reset watchdog
                        frame = np.frombuffer(
                            raw, dtype=np.uint8
                        ).reshape((dec_h, dec_w, 3)).copy()
                        with self._lock:
                            self._frame_q = [frame]   # keep only latest
                            self._last_frame_ts = time.time()
                        if not hasattr(self, '_ffl'):
                            self._ffl = True
                            print('[RTSP] ✅ SUCCESS: First frame decoded and ready!')
                    else:
                        if time.time() - last_data[0] > WATCHDOG:
                            print('[RTSP] frame_reader: empty read watchdog — restarting')
                            proc.kill()
                            break
                except Exception as _e:
                    print(f'[RTSP] frame_reader error: {_e}')
                    break

        threading.Thread(target=frame_reader, daemon=True).start()

        # Global NAL buffer across RTP packets
        h264_buf = bytearray()
        nal_buf  = bytearray()
        buf      = b''

        # ── Send SPS/PPS first ──────────────────────────────────────
        # Crucial for ffmpeg to start decoding immediately
        if spspps:
            try:
                proc.stdin.write(spspps)
                proc.stdin.flush()
                print(f'[RTSP] Injected SPS/PPS ({len(spspps)} bytes) into decoder')
            except Exception as e:
                print(f'[RTSP] Failed to inject SPS/PPS: {e}')

        while self._running:
            try:
                self._tls_sock.settimeout(5)
                chunk = self._tls_sock.recv(131072)
                if not chunk:
                    break
                buf += chunk

                while len(buf) >= 4:
                    if buf[0] != 0x24:
                        idx = buf.find(b'$')
                        buf = buf[idx:] if idx >= 0 else b''
                        break
                    ln = struct.unpack('!H', buf[2:4])[0]
                    if len(buf) < 4 + ln:
                        break
                    ch = buf[1]; rtp = buf[4:4+ln]; buf = buf[4+ln:]
                    # Process video channels (usually 0, 2, or 4)
                    if ch % 2 != 0 or len(rtp) < 13:
                        continue

                    payload = rtp[12:]
                    marker  = (rtp[1] & 0x80) != 0
                    nt      = payload[0] & 0x1F

                    if nt == 28:   # FU-A fragmented
                        if len(payload) < 2: continue
                        fu = payload[1]
                        if fu & 0x80: # Start bit
                            nal_buf = bytearray([(payload[0] & 0xE0) | (fu & 0x1F)])
                        nal_buf.extend(payload[2:])
                        if fu & 0x40: # End bit
                            h264_buf.extend(b'\x00\x00\x00\x01' + nal_buf)
                            nal_buf = bytearray()
                    elif nt == 24:  # STAP-A (aggregated)
                        off = 1
                        while off + 2 <= len(payload):
                            sz = struct.unpack('!H', payload[off:off+2])[0]; off += 2
                            h264_buf.extend(b'\x00\x00\x00\x01' + payload[off:off+sz])
                            off += sz
                    else: # Single NAL
                        h264_buf.extend(b'\x00\x00\x00\x01' + payload)

                    # Flush to ffmpeg if we have any NALs or the marker bit is set
                    if len(h264_buf) > 0:
                        try:
                            proc.stdin.write(bytes(h264_buf))
                            proc.stdin.flush()
                            h264_buf.clear()
                        except Exception:
                            break

            except Exception as e:
                if self._running:
                    print('[RTSP] RTP reader: ' + str(e))
                break

        # Flush remaining
        try:
            if h264_buf:
                proc.stdin.write(bytes(h264_buf))
            proc.stdin.close()
            proc.kill()
        except Exception:
            pass

    def _decode_nal(self, nal_data: bytes):
        pass

    def _try_ffmpeg(self) -> bool:
        """Plain ffmpeg pipe."""
        return self._launch_ffmpeg(self._url, tls=False)

    def _launch_ffmpeg(self, url: str, tls: bool) -> bool:
        """Launch ffmpeg subprocess pipe and verify first frame."""
        cmd = ['ffmpeg', '-loglevel', 'error']
        if tls:
            cmd += ['-tls_verify', '0']
        cmd += [
            '-rtsp_transport', 'tcp',
            '-i', url,
            '-vf', f'scale={self._w}:{self._h}',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-an', '-sn', 'pipe:1'
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=10**8)
            frame_size = self._w * self._h * 3
            raw = proc.stdout.read(frame_size)
            if len(raw) == frame_size:
                self._proc    = proc
                self._buf     = raw
                self._opened  = True
                print(f"[RTSP] ✅ ffmpeg pipe {'TLS' if tls else 'plain'} — "
                      f"{self._w}×{self._h}")
                return True
            err = proc.stderr.read(300).decode('utf-8', errors='ignore')
            print(f"[RTSP] ffmpeg {'TLS' if tls else 'plain'} failed: "
                  f"{err[-100:]}")
            proc.kill()
        except Exception as e:
            print(f"[RTSP] ffmpeg error: {e}")
        return False

    def isOpened(self):
        if self._tls_sock is not None:
            return self._opened
        return self._opened and self._proc is not None

    def read(self):
        if not self.isOpened():
            return False, None
        # Both TLS and direct paths store frames in _frame_q
        with self._lock:
            if self._frame_q:
                frame = self._frame_q[-1]
                self._frame_q = []   # clear so next read() returns False until new frame
                return True, frame.copy()
        return False, None

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:          return 25.0
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:  return float(self._w)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT: return float(self._h)
        return 0.0

    def set(self, prop_id, value):
        return False

    def release(self):
        self._running = False
        if self._tls_sock:
            try: self._tls_sock.close()
            except Exception: pass
            self._tls_sock = None
        if self._proc:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception: pass
            self._proc = None
        self._opened = False


class FFmpegVideoCapture:
    def isOpened(self):
        return True # Process is managed internally
    def release(self):
        pass

    def __init__(self, url, width=1280, height=960):   # FIXED: was height=720
        self.url = url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        
        import threading
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
    def _capture_loop(self):
        import subprocess
        import numpy as np
        import time
        import select

        # Pure software H264 decode — correct BGR24 output.
        # hwaccel REMOVED: v4l2m2m output format caused BGR conversion errors.
        # probesize/analyzeduration NOT restricted (were 32/0 → decoder init fail).
        cmd_base = ['ffmpeg',
                    '-thread_queue_size', '64',
                    '-fflags',            'nobuffer',
                    '-flags',             'low_delay',
                    '-rtsp_transport',    'tcp']

        cmd_tail = [
            '-i',       self.url,
            '-f',       'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s',       f'{self.width}x{self.height}',
            '-r',       '20',
            '-an', '-sn', '-dn',
            'pipe:1',
        ]

        cmd = cmd_base + cmd_tail

        # Pre-allocate read buffer
        _buf = bytearray(self.frame_size)
        _mv  = memoryview(_buf)
        
        while self.running:
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE,
                    # CHANGED from DEVNULL → PIPE so FFmpeg errors are visible
                    stderr=subprocess.PIPE,
                    bufsize=self.frame_size)
                
                # Drain stderr in background so it never blocks stdout
                def _drain(p):
                    try:
                        for raw in p.stderr:
                            line = raw.decode('utf-8', errors='ignore').strip()
                            if line:
                                print(f'[CAM1-FFMPEG] {line}')
                    except Exception:
                        pass
                threading.Thread(target=_drain, args=(proc,), daemon=True).start()

            except Exception as e:
                print(f"[FFmpeg Capture] Failed to start: {e}")
                time.sleep(1)
                continue
                
            failures = 0
            while self.running:
                try:
                    # REDUCED: 2 s (was 3 s) — detect dropout faster
                    readable, _, _ = select.select([proc.stdout], [], [], 2.0)
                    if not readable:
                        print("[FFmpeg Capture] Read timed out — reconnecting")
                        break
                    n = 0
                    while n < self.frame_size:
                        got = proc.stdout.readinto(_mv[n:])
                        if not got:
                            break
                        n += got
                except Exception:
                    failures += 1
                    n = 0
                    
                if n != self.frame_size:
                    failures += 1
                    # REDUCED: 5 bad frames (was 10) — reconnect faster
                    if failures >= 5:
                        print("[FFmpeg Capture] Too many bad frames — restarting")
                        break
                    time.sleep(0.01)
                    continue
                    
                failures = 0
                frame = np.frombuffer(_buf, dtype=np.uint8
                                     ).reshape((self.height, self.width, 3)).copy()
                with self.lock:
                    self.latest_frame = frame
                    self._last_frame_ts = time.time()
                    
            try:
                proc.kill()
                proc.stdout.close()
            except:
                pass
            
            if self.running:
                # REDUCED: 0.5 s (was 1.0 s) — faster reconnect
                time.sleep(0.5)

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            if time.time() - getattr(self, '_last_frame_ts', time.time()) > 20.0:
                return False, None
            return True, self.latest_frame.copy()
            
    def release(self):
        self.running = False


def generate_frames():
    """
    Dual pipeline for RTSP cameras:
    - FFmpeg Hardware decodes to BGR directly into a shared variable.
    - _stream runs the reader/encoder/infer logic using the latest frame.
    """
    if state.video_source:
        source_str = str(state.video_source)
        print(f"[STREAM] Initializing video source: {source_str}")
        is_rtsp = source_str.startswith('rtsp://')
        if is_rtsp:
            cap = _open_rtsp(source_str)
            yield from _stream(cap, native_fps=25.0)
            return
        if os.path.exists(source_str):
            cap = cv2.VideoCapture(source_str)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            yield from _stream(cap, native_fps=fps)
            return

    # Local camera
    cam_id = getattr(state, 'local_camera_id', 0)
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    yield from _stream(cap, native_fps=None)


def _stream_rtsp_direct(rtsp_url: str):
    """
    Real-time RTSP streaming with Frame ID sync.
    Each frame gets a unique ID. Inference result is bound to
    the exact frame it was computed from. Streamer draws the
    overlay that belongs to the frame being shown.
    """
    my_gen = state.stream_generation

    # Parse URL
    import re as _re
    m = _re.match(r'rtsp://([^:@]+):([^@]*)@([^:/]+):?(\d*)(.*)', rtsp_url)
    if not m:
        print(f'[STREAM] Invalid RTSP URL: {rtsp_url}')
        return
    user, pwd, host, port, path = m.groups()

    # Start TLS proxy
    _start_tls_proxy(host, 443, 5554)
    import socket as _ck
    for _ in range(30):
        try:
            t = _ck.create_connection(('127.0.0.1', 5554), timeout=0.1)
            t.close(); break
        except: time.sleep(0.1)

    proxy_url = f'rtsp://{user}:{pwd}@127.0.0.1:5554{path}'

    # ── Frame store: {frame_id: jpeg_bytes} ──────────────────────
    import threading as _th
    frame_lock    = _th.Lock()
    frame_id      = [0]          # monotonic counter
    frame_store   = {}           # frame_id → jpeg bytes
    overlay_store = {}           # frame_id → annotated jpeg bytes
    MAX_STORE     = 5            # keep last N frames only

    # ── READER: ffmpeg → parse JPEG → store with ID ──────────────
    stream_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-fflags', 'nobuffer+discardcorrupt',
        '-flags', 'low_delay',
        '-rtsp_transport', 'tcp',
        '-i', proxy_url,
        '-vf', 'scale=1280:720',   # consistent resolution — prevents artifact lines
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        '-q:v', '3',               # quality 3 (1=best, 31=worst) — removes JPEG artifacts
        'pipe:1'
    ]

    def _start_ffmpeg():
        try:
            proc = subprocess.Popen(
                stream_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0)
            # Drain stderr to prevent pipe deadlock
            import threading as _th2
            def _drain(p):
                try:
                    while p.poll() is None:
                        p.stderr.readline()
                except Exception: pass
            _th2.Thread(target=_drain, args=(proc,), daemon=True).start()
            return proc
        except Exception as e:
            print(f'[STREAM] ffmpeg start failed: {e}')
            return None

    stream_proc = _start_ffmpeg()
    if stream_proc is None:
        return

    def jpeg_reader():
        nonlocal stream_proc
        buf          = b''
        last_data_ts = [time.time()]
        WATCHDOG_SEC = 15.0  # Increased from 5.0s to handle heavy CPU load

        while state.camera_active and state.stream_generation == my_gen:
            try:
                import select as _sel
                rdy, _, _ = _sel.select([stream_proc.stdout], [], [], 1.0)
                if not rdy:
                    if time.time() - last_data_ts[0] > WATCHDOG_SEC:
                        print('[STREAM] Watchdog timeout — restarting ffmpeg')
                        try: stream_proc.kill()
                        except: pass
                        time.sleep(1.0)
                        stream_proc = _start_ffmpeg()
                        if stream_proc is None: break
                        buf = b''
                        last_data_ts[0] = time.time()
                    continue

                chunk = stream_proc.stdout.read(65536)
                if not chunk:
                    print('[STREAM] ffmpeg pipe closed — restarting')
                    try: stream_proc.kill()
                    except: pass
                    time.sleep(1.0)
                    stream_proc = _start_ffmpeg()
                    if stream_proc is None: break
                    buf = b''
                    last_data_ts[0] = time.time()
                    continue

                last_data_ts[0] = time.time()
                buf += chunk

                while True:
                    s = buf.find(b'\xff\xd8')
                    if s < 0: buf = b''; break
                    e = buf.find(b'\xff\xd9', s + 2)
                    if e < 0: buf = buf[s:]; break
                    jpeg = buf[s:e + 2]
                    buf  = buf[e + 2:]
                    with frame_lock:
                        fid = frame_id[0] + 1
                        frame_id[0] = fid
                        frame_store[fid] = jpeg
                        for old in list(frame_store.keys()):
                            if old < fid - MAX_STORE:
                                frame_store.pop(old, None)
                                overlay_store.pop(old, None)
            except Exception as _ex:
                print(f'[STREAM] jpeg_reader error: {_ex}')
                time.sleep(0.5)

        try: stream_proc.kill()
        except: pass
        print('[STREAM] jpeg_reader exited')

    _th.Thread(target=jpeg_reader, daemon=True).start()

    # ── INFER: decode JPEG → Hailo → build overlay → store with same ID ─
    def infer_thread():
        last_inferred = [0]
        while state.camera_active and state.stream_generation == my_gen:
            try:
                # Get latest unprocessed frame
                with frame_lock:
                    fid  = frame_id[0]
                    jpeg = frame_store.get(fid)

                if jpeg is None or fid == last_inferred[0]:
                    time.sleep(0.04)
                    continue

                last_inferred[0] = fid

                # Decode to BGR
                arr   = np.frombuffer(jpeg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Run inference — updates state, returns annotated frame
                h_fr, w_fr = frame.shape[:2]
                state.orig_frame = frame.copy()
                state._scale_x   = 1.0
                state._scale_y   = 1.0
                detections = detect_hand(frame)
                result = process_frame(frame, detections)

                # Store clean frame for capture
                if state.hand_bbox is None and state.panel_rect is not None:
                    state.last_clean_frame      = frame.copy()
                    state.last_clean_panel_rect = state.panel_rect
                    state.last_clean_contour    = state.panel_contour
                    state.last_clean_serial_det = state.serial_det

                # Draw overlay on THIS exact frame — bind result to frame ID
                annotated = result if result is not None else draw_overlay(frame.copy(), state.current_sequence, getattr(state, 'status_msg', ''), getattr(state, 'progress', 0))
                _, buf = cv2.imencode('.jpg', annotated,
                                      [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                with frame_lock:
                    overlay_store[fid] = buf.tobytes()
                    # Evict old overlays
                    for old in list(overlay_store.keys()):
                        if old < fid - MAX_STORE:
                            overlay_store.pop(old, None)

                # ~5fps inference rate
                time.sleep(0.2)

            except Exception as e:
                print(f'[INFER] {e}')
                time.sleep(0.1)

    _th.Thread(target=infer_thread, daemon=True).start()

    # ── STREAMER: serve latest frame — no freeze, no flicker ─────
    MIN_INTERVAL   = 1.0 / STREAM_FPS_CAP
    last_streamed_fid = [0]
    last_overlay   = [None]
    last_overlay_fid = [0]

    while state.camera_active and state.stream_generation == my_gen:
        with frame_lock:
            fid = frame_id[0]
            raw = frame_store.get(fid)
            # Update last overlay if new one available
            for check in range(fid, max(0, fid - MAX_STORE), -1):
                if check in overlay_store:
                    if check > last_overlay_fid[0]:
                        last_overlay[0]     = overlay_store[check]
                        last_overlay_fid[0] = check
                    break

        if raw is None:
            time.sleep(0.005)
            continue

        # Always stream latest raw frame — overlay applied on top if available
        if fid == last_streamed_fid[0]:
            time.sleep(0.005)
            continue

        last_streamed_fid[0] = fid

        # Use overlay if it's recent (within 10 frames), else raw
        overlay_fresh = (fid - last_overlay_fid[0]) <= 10
        jpeg = last_overlay[0] if (last_overlay[0] is not None and overlay_fresh) else raw

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + jpeg + b'\r\n')

        time.sleep(MIN_INTERVAL)

    try: stream_proc.kill()
    except: pass


# ─────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────

# Camera-2 (OCR) URL — set by /api/start_camera, read by /api/video_feed_cam2
_camera2_url = None

# Camera-2 OCR engine instance (parallel thread — no Camera-1 impact)
camera2_ocr_instance = None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    global _camera2_url
    data        = request.json or {}
    rtsp_url    = data.get('rtsp_url', '').strip()
    device_id   = data.get('device_id', 0)
    camera2_url = data.get('camera2_url', '').strip()   # OCR camera (UI only)

    state.stream_generation += 1
    state.camera_active      = False
    time.sleep(0.15)

    if rtsp_url:
        state.video_source    = rtsp_url
        msg = f'LAN Camera — {rtsp_url[:40]} — {state.backend.upper()}'
    else:
        state.video_source    = None
        state.local_camera_id = int(device_id) if str(device_id).isdigit() else 0
        msg = f'Local Camera {state.local_camera_id} — {state.backend.upper()}'

    # Store Camera-2 URL (OCR overlay) — no inference done here
    _camera2_url = camera2_url if camera2_url else None

    # ── Camera-2 OCR instance (ADD ONLY) ─────────────────────────────
    global camera2_ocr_instance
    if _CAM2_OCR_AVAILABLE and camera2_url:
        # Give the main Hailo engine a moment to settle if it just started
        time.sleep(2.0)

        # Stop previous instance cleanly before creating a new one
        if camera2_ocr_instance is not None:
            try:
                camera2_ocr_instance.stop()
            except Exception:
                pass
        def _on_cam2_serial(serial):
            print(f'[CAM2-OCR] Callback → serial={serial}')
            if serial and serial not in (None, 'UNKNOWN', 'Searching...', 'Reading...'):
                # FIX A2: panel_id is None means reset_panel() already ran.
                # This is a late callback for the OLD panel — do NOT create a
                # new folder for the NEXT panel before its first image exists.
                if state.panel_id is None:
                    print(f'[CAM2-OCR] Late serial ignored (panel already reset): {serial}')
                    return
                state.serial_number = serial
                state.ocr_done      = True
                folder = state.current_sequence_panel_folder
                if folder:
                    threading.Thread(target=rename_panel_images, args=(folder, serial), daemon=True).start()

        # ── Resolve serial.pt path absolutely so it works regardless
        # of the process CWD (Flask can be launched from any directory).
        _script_dir       = os.path.dirname(os.path.abspath(__file__))
        _serial_pt_path   = r"D:\SSP\models\serial.pt"
        if not os.path.exists(_serial_pt_path):
            # Fallback: models/ beside this script file
            _serial_pt_path = os.path.join(_script_dir, "models", "serial.pt")
        print(f'[CAM2-OCR] serial.pt path → {_serial_pt_path}  '
              f'exists={os.path.exists(_serial_pt_path)}')

        camera2_ocr_instance = Camera2OCR(
            camera2_url=camera2_url,
            open_cap_fn=lambda u: _open_rtsp(u, width=2048, height=1536),
            on_serial_detected=_on_cam2_serial,
            pt_path=_serial_pt_path
        )
        print(f'[CAM2-OCR] Instance ready for: {camera2_url[:60]}')
    else:
        camera2_ocr_instance = None

    state.camera_active = True
    return jsonify({
        'success':     True,
        'backend':     state.backend,
        'source':      'rtsp' if rtsp_url else 'camera',
        'message':     msg,
        'stream_gen':  state.stream_generation,
        'cam2_active': bool(_camera2_url),
    })


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """
    Accept a video upload and stream it straight to disk in 1 MB chunks.
    This works for any file size regardless of available RAM.
    """
    upload_dir = os.path.join(BASE_STORAGE, '_uploads')
    os.makedirs(upload_dir, exist_ok=True)

    # ── Get filename from Content-Disposition or fallback ─────
    cd          = request.headers.get('Content-Disposition', '')
    fname       = 'upload.mp4'
    for part in cd.split(';'):
        part = part.strip()
        if part.startswith('filename='):
            fname = part.split('=', 1)[1].strip(' "\'')
            break

    # Use form-data filename if available (multipart upload)
    if 'video' in request.files:
        f         = request.files['video']
        safe_name = datetime.now().strftime("%Y%m%d_%H%M%S_") + \
                    os.path.basename(f.filename or fname)
        save_path = os.path.join(upload_dir, safe_name)
        # Stream to disk in 1 MB chunks
        CHUNK = 1024 * 1024
        with open(save_path, 'wb') as out:
            while True:
                chunk = f.stream.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)

    elif request.content_length or request.stream:
        # Raw body stream (application/octet-stream)
        safe_name = datetime.now().strftime("%Y%m%d_%H%M%S_") + fname
        save_path = os.path.join(upload_dir, safe_name)
        CHUNK     = 1024 * 1024
        with open(save_path, 'wb') as out:
            while True:
                chunk = request.stream.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
    else:
        return jsonify({'success': False, 'error': 'No video data received'}), 400

    # Verify file was saved and is readable
    if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
        return jsonify({'success': False, 'error': 'File save failed'}), 500

    file_mb = os.path.getsize(save_path) / (1024 * 1024)

    # Probe FPS and frame count
    cap          = cv2.VideoCapture(save_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Stop old stream and start fresh for new video
    state.camera_active      = False
    time.sleep(0.1)
    state.stream_generation += 1
    state.video_source        = save_path
    state.camera_active       = True
    state.prev_gray           = None
    state.hand_bbox           = None
    state.reset_for_new_panel()

    print(f"[INFO] Video saved: {safe_name} | {file_mb:.1f} MB | "
          f"{fps:.1f} FPS | {total_frames} frames")

    return jsonify({
        'success'     : True,
        'source'      : 'video',
        'backend'     : state.backend,
        'fps'         : round(fps, 2),
        'total_frames': total_frames,
        'file_mb'     : round(file_mb, 1),
        'filename'    : safe_name,
        'message'     : f'Video ready — {fps:.1f} FPS real-time playback'
    })


@app.route('/api/list_cameras', methods=['GET'])
def list_cameras():
    """
    Probe cv2.VideoCapture indices 0–4 to find cameras physically connected to the Pi.
    Returns a list like: [{"index": 0, "label": "Camera 0 — 1280×720"}]
    This is completely different from browser mediaDevices which lists the PC's cameras.
    """
    found = []
    for idx in range(5):
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    found.append({
                        'index': idx,
                        'label': f'Camera {idx}  —  {w}×{h}'
                    })
            cap.release()
        except Exception:
            pass
    if not found:
        found = [{'index': 0, 'label': 'Camera 0 (default)'}]
    print(f"[INFO] Cameras on Pi: {found}")
    return jsonify({'cameras': found})


@app.route('/api/test_connection', methods=['POST'])
def test_connection():
    """Test camera connection — waits for first frame from TLS stream."""
    data     = request.json or {}
    rtsp_url = data.get('rtsp_url', '').strip()
    if not rtsp_url:
        return jsonify({'success': False, 'error': 'No URL provided'}), 400
    try:
        cap = _open_rtsp(rtsp_url)
        if not cap.isOpened():
            return jsonify({'success': False,
                            'error': 'Cannot connect — check IP, port, credentials'})

        # Wait up to 12s for first frame (TLS stream needs time to decode first keyframe)
        import time as _time
        ok = False; frame = None
        for _ in range(120):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            _time.sleep(0.1)

        cap.release()
        if not ok or frame is None:
            return jsonify({'success': False,
                            'error': 'Connected but no video frame received — camera may be off or stream path wrong'})
        fh, fw = frame.shape[:2]
        return jsonify({
            'success': True,
            'width':   fw,
            'height':  fh,
            'fps':     25.0,
            'message': f'Connected — {fw}×{fh} @ 25 FPS'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    state.camera_active = False
    state.video_source  = None
    return jsonify({'success': True})


@app.route('/api/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def _generate_cam2_frames():
    """
    Camera-2 UI stream.
    Pulls frames from the OCR engine's _cap directly — always fresh.
    Auto-recovers if frame goes stale.
    """
    last_frame_ts    = [time.time()]
    last_frame_bytes = [None]
    STREAM_WATCHDOG  = 5.0
    STREAM_FPS       = 8

    def _make_banner(text, color_bgr=(60,60,60)):
        blank = np.zeros((960, 1280, 3), dtype=np.uint8)
        cv2.rectangle(blank, (0, 0), (1280, 960), color_bgr, 8)
        cv2.putText(blank, text, (60, 480),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, color_bgr, 3)
        _, b = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 50])
        return b.tobytes()

    while True:
        if not _camera2_url or not camera2_ocr_instance:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + _make_banner("CAMERA 2 NOT CONFIGURED") + b'\r\n')
            time.sleep(0.5)
            continue

        try:
            # Use get_annotated_frame() so the serial.pt bounding box is
            # always drawn on the Camera-2 stream — operators see the green
            # "serial" bbox the moment the model detects the serial region.
            frame = camera2_ocr_instance.get_annotated_frame()

            if frame is not None:
                last_frame_ts[0] = time.time()

                # ── Always rebuild the display frame on every loop iteration ──
                # The old hash-based check (`if fhash != last_frame_hash[0]:`)
                # prevented updates when the camera scene was static.
                # Consequence: the serial.pt bbox AND the OCR status text
                # (Scanning / Serial Captured) were NEVER updated once the
                # hash stabilised — so detection annotations were invisible
                # on any non-moving scene.
                # Fix: always encode and serve the latest annotated frame.
                display   = frame.copy()
                is_done   = camera2_ocr_instance.is_done()
                scanning  = getattr(camera2_ocr_instance, '_is_scanning', False)
                serial    = camera2_ocr_instance.get_serial_number()
                status    = getattr(camera2_ocr_instance, 'status', '')
                yolo_det   = camera2_ocr_instance.get_latest_detection()
                yolo_ready = camera2_ocr_instance.serial_detector is not None

                # ── HEF-not-loaded warning overlay ────────────────────────
                # If serial_detector is None (HEF file missing or VDevice
                # error), draw a persistent red banner so the operator knows
                # annotations are disabled instead of silently seeing a clean
                # frame with no explanation.
                if not yolo_ready:
                    h_f, w_f = display.shape[:2]
                    cv2.rectangle(display, (0, h_f - 40), (w_f, h_f),
                                  (0, 0, 180), -1)
                    cv2.putText(display, "serial.pt NOT LOADED — no annotation",
                                (10, h_f - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2, cv2.LINE_AA)

                # ── Status text — top-left corner ─────────────────────────
                if is_done and serial and serial != "Reading...":
                    status_txt = f"Serial: {serial}  ✓"
                    txt_color  = (0, 220, 80)    # green
                elif scanning and yolo_det is not None:
                    status_txt = "Scanning — serial detected"
                    txt_color  = (0, 220, 255)   # yellow
                elif scanning:
                    status_txt = status or "Scanning..."
                    txt_color  = (0, 180, 255)   # amber
                else:
                    status_txt = ""
                    txt_color  = (180, 180, 180)

                if status_txt:
                    (tw, th), _ = cv2.getTextSize(
                        status_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                    pad = 8
                    cv2.rectangle(display,
                                  (10, 8),
                                  (10 + tw + pad*2, 8 + th + pad*2),
                                  (20, 20, 20), -1)
                    cv2.putText(display, status_txt,
                                (10 + pad, 8 + th + pad),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                txt_color, 2, cv2.LINE_AA)

                _, jpg = cv2.imencode('.jpg', display,
                                     [cv2.IMWRITE_JPEG_QUALITY, 70])
                last_frame_bytes[0] = jpg.tobytes()

                # Serve latest frame (fresh or cached)
                age = time.time() - last_frame_ts[0]
                if age > STREAM_WATCHDOG:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + _make_banner("CAMERA 2 RECONNECTING...",
                                          (0, 80, 200)) + b'\r\n')
                    time.sleep(0.5)
                    continue

                if last_frame_bytes[0]:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + last_frame_bytes[0] + b'\r\n')
            else:
                age = time.time() - last_frame_ts[0]
                if age > STREAM_WATCHDOG:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + _make_banner("CAMERA 2 CONNECTING...",
                                          (0, 100, 255)) + b'\r\n')
                time.sleep(0.3)
                continue

        except Exception as e:
            print(f'[CAM2 STREAM] {e}')
            time.sleep(0.5)
            continue

        time.sleep(1.0 / STREAM_FPS)



@app.route('/api/video_feed_cam2')
def video_feed_cam2():
    """Camera-2 OCR overlay stream — raw MJPEG passthrough, no inference."""
    return Response(_generate_cam2_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cam2_status')
def cam2_status():
    """Camera-2 OCR state — polled by UI every 500ms for yellow/green/red."""
    if not _CAM2_OCR_AVAILABLE or camera2_ocr_instance is None:
        return jsonify({'available': False, 'state': 'offline',
                        'serial': None, 'partial': None, 'attempts': 0,
                        'ocr_status': 'Camera 2 offline'})
    done       = camera2_ocr_instance.is_done()
    serial     = camera2_ocr_instance.get_serial_number()
    partial    = getattr(camera2_ocr_instance, 'partial_serial', None)
    scanning   = getattr(camera2_ocr_instance, '_is_scanning', False)
    attempts   = getattr(camera2_ocr_instance, 'ocr_attempt_count', 0)
    ocr_status = getattr(camera2_ocr_instance, 'status', '')

    # Phase-2 in progress: frame captured but serial not yet read
    phase2_reading = done and serial == "Reading..."

    # Update main state serial only once Phase-2 produces a real serial
    if done and serial and serial != "Reading..." \
            and (state.serial_number in (None, 'UNKNOWN', 'Searching...', 'Reading...')):
        state.serial_number = serial
        state.ocr_done      = True
        print(f"[CAM2] Serial updated from Phase-2: {serial}")
        
        # Rename existing images in the folder to include the serial
        folder = ensure_panel_folder()
        rename_panel_images(folder, serial)
        
        # Auto-complete SEQ1 once serial is captured
        if not state.completed.get(1):
            print(f"[CAM2] Auto-completing SEQ1 (Serial Captured)")
            complete(1)

    # Sync partial serial to state for sequence_status API
    if partial and partial not in ('Reading...', 'UNKNOWN'):
        state.partial_serial = partial

    # Determine the state key the UI understands
    if done and not phase2_reading:
        ui_state = 'found'
    elif done and phase2_reading:
        ui_state = 'captured'   # green immediately, serial still being read
    elif scanning:
        ui_state = 'scanning'
    else:
        ui_state = 'idle'

    return jsonify({
        'available': True,
        'state':     ui_state,
        'serial':    serial,
        'partial':   partial,
        'attempts':  attempts,
        'ocr_status': ocr_status,
    })


@app.route('/api/sequence_status', methods=['GET'])
def sequence_status():
    # Only refresh date (not directory count) — count is maintained in memory
    new_date = datetime.now().strftime("%Y-%m-%d")
    if new_date != state.today_date:
        state.today_date        = new_date
        state.panel_count_today = 0   # midnight reset
        # Persist reset
        try:
            count_file = os.path.join(BASE_STORAGE, 'panel_count.txt')
            with open(count_file, 'w') as _f:
                _f.write(f"{new_date},0")
        except Exception:
            pass

    # 5. NORMALIZE SEQUENCE STATUS (FIX UI BUG)
    model_sees_seq = getattr(state, 'model_sees', 'NONE')
    for i in [1, 2, 3]:
        if getattr(state, 'completed', {}).get(i):
            # ── Sequence is done — keep whatever final status was set ──
            # completed[i]=True is set by complete() which sets 'completed',
            # AND also by the missed path which sets 'missed'. So read the
            # real sequence_status[i] rather than forcing 'completed'.
            # If it hasn't been set to 'missed' explicitly, default to 'completed'.
            if state.sequence_status.get(i) not in ('completed', 'missed'):
                state.sequence_status[i] = 'completed'
            # (else: keep existing 'completed' or 'missed' — do not overwrite)
        elif i == state.current_sequence:
            # Use BOTH wiping_active flag AND status_msg check so the UI badge
            # never lags a poll cycle behind the actual wipe state.
            is_wiping = (state.wiping_active or
                         "WIPING" in getattr(state, 'status_msg', ""))
            state.sequence_status[i] = "wiping" if is_wiping else "active"
        elif state.sequence_status.get(i) == 'missed':
            # Preserve 'missed' — don't let poll cycle overwrite it to 'pending'
            pass
        elif (state.current_sequence == 0
              and not state.all_sequences_done
              and not state.completed.get(i, False)
              and model_sees_seq == f"panel_seq{i}"
              and getattr(state, 'panel_absent_frames', 99) < 5):
            # IDLE + model actively sees panel live — give operator live feedback.
            # FIX B3: guards prevent ghost badge after panel removal.
            state.sequence_status[i] = "active"
        else:
            state.sequence_status[i] = "pending"

    # Build per-sequence cleaning percentage
    per_seq_pct = {}
    for i in (1, 2, 3):
        if state.sequence_status.get(i) == 'completed':
            per_seq_pct[str(i)] = 100.0
        elif i == state.current_sequence:
            per_seq_pct[str(i)] = state.progress
        else:
            per_seq_pct[str(i)] = 0.0

    # Convert sequence_status to string keys for JSON compatibility
    seq_status_str = {str(k): v for k, v in state.sequence_status.items()}

    # ── FIX TOTAL LEAD TIME ───────────────────────
    total_time = 0
    if getattr(state, 'panel_end_time', None) and getattr(state, 'panel_start_time', None):
        total_time = state.panel_end_time - state.panel_start_time
    elif getattr(state, 'panel_start_time', None):
        total_time = time.time() - state.panel_start_time

    seq_times = {
        1: (state.seq_end_time.get(1) or time.time()) - (state.seq_start_time.get(1) or time.time()) if state.seq_start_time.get(1) else 0,
        2: (state.seq_end_time.get(2) or time.time()) - (state.seq_start_time.get(2) or time.time()) if state.seq_start_time.get(2) else 0,
        3: (state.seq_end_time.get(3) or time.time()) - (state.seq_start_time.get(3) or time.time()) if state.seq_start_time.get(3) else 0,
    }

    # ── Landscape alert lifecycle ─────────────────────────────────
    # Detect SEQ2/SEQ3 first-capture and trigger "captured" voice + green flash
    if (state.landscape_alert == "seq2"
            and state.seq2_auto_captured
            and not state._seq2_cap_announced):
        state._seq2_cap_announced = True
        state.landscape_alert     = "captured"
        state.landscape_alert_ts  = time.time()
        speak("S E Q 2 captured. Good job!", lang='en')
        def _hi_cap2():
            import shutil
            exe = ('espeak-ng' if shutil.which('espeak-ng')
                   else 'espeak' if shutil.which('espeak') else None)
            if exe:
                try:
                    subprocess.run([exe, '-v', 'hi', '-s', '120', '-a', '90',
                                    'SEQ do. Photo le liya. Shukriya'],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=6)
                except Exception:
                    pass
        threading.Thread(target=_hi_cap2, daemon=True).start()

    elif (state.landscape_alert == "seq3"
            and state.seq3_auto_captured
            and not state._seq3_cap_announced):
        state._seq3_cap_announced = True
        state.landscape_alert     = "captured"
        state.landscape_alert_ts  = time.time()
        speak("S E Q 3 captured. Good job!", lang='en')
        def _hi_cap3():
            import shutil
            exe = ('espeak-ng' if shutil.which('espeak-ng')
                   else 'espeak' if shutil.which('espeak') else None)
            if exe:
                try:
                    subprocess.run([exe, '-v', 'hi', '-s', '120', '-a', '90',
                                    'SEQ teen. Photo le liya. Shukriya'],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=6)
                except Exception:
                    pass
        threading.Thread(target=_hi_cap3, daemon=True).start()

    # Auto-clear "captured" green banner after 3 seconds
    elif (state.landscape_alert == "captured"
            and time.time() - state.landscape_alert_ts > 3.0):
        state.landscape_alert = ""

    return jsonify({
        "sequence": state.current_sequence,
        "status": seq_status_str,
        # Return '' for UI until a real serial is confirmed — resets blank on new panel
        "serial": state.serial_number
                  if state.serial_number not in (None, 'UNKNOWN', 'Searching...', 'Reading...', '')
                  else '',
        "reset_id": state.panel_reset_id,
        'seq_time': {str(k): v for k, v in seq_times.items()},
        'total_time': total_time,

        # Keep existing fields so nothing breaks
        'current_sequence':    state.current_sequence,
        'sequence_status':     seq_status_str,
        'cleaning_percentage': state.progress,
        'per_seq_pct':         per_seq_pct,
        # Blank for UI header until confirmed; '' resets UI on new panel
        'serial_number':       state.serial_number
                               if state.serial_number not in (None, 'UNKNOWN', 'Searching...', 'Reading...', '')
                               else '',
        'captures':            {str(k): v for k, v in state.sequence_captured.items()},
        'backend':             state.backend,
        'panel_count_today':   state.panel_count_today,
        'today_date':          state.today_date,
        # Show partial only while scanning; hide once full serial confirmed
        'partial_serial':      (state.partial_serial
                                if not state.ocr_done
                                else None)
                               or (getattr(camera2_ocr_instance, 'partial_serial', None)
                                   if camera2_ocr_instance and not state.ocr_done
                                   else None),
        'wiping_active':       state.wiping_active,
        'current_panel_name':  getattr(state, 'current_panel_name', state.current_sequence_panel_name),
        'all_sequences_done':  state.all_sequences_done,
        'panel_reset_id':      state.panel_reset_id,
        'failure_reason':      getattr(state, 'failure_reason', None),
        # ── Landscape placement alert (voice + UI colour) ──────────
        # "" = nothing | "seq2" = orange | "seq3" = blue | "captured" = green
        'landscape_alert':     getattr(state, 'landscape_alert', ''),
        # Camera-2 fields — needed by UI auto-reconnect on page refresh
        'cam2_active':         bool(_camera2_url),
        'cam2_url':            _camera2_url or '',
        'lead_times': {
            str(i): round(
                (state.seq_end_time.get(i) or time.time()) -
                (state.seq_start_time.get(i) or time.time()), 1)
            if state.seq_start_time.get(i) else None
            for i in (1, 2, 3)
        },
        'total_lead_time': round(
            ((state.panel_end_time or time.time()) -
             (state.panel_start_time or time.time())), 1)
        if state.panel_start_time else None,
    })


@app.route('/api/reset', methods=['POST'])
def reset_system():
    state.reset_for_new_panel()
    return jsonify({'success': True, 'message': 'System reset'})


@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Shows raw model output to diagnose class name issues."""
    return jsonify({
        'current_sequence':     state.current_sequence,
        'sequence_status':      state.sequence_status,
        'cleaning_percentage':  state.progress,
        'current_panel_name':   getattr(state, 'current_sequence_panel_name', None),
        'serial_number':        state.serial_number,
        'detected_seq_history': list(state.detected_seq_history)[-10:],
        'wiping_active':        state.wiping_active,
        'no_wiping_frames':     state.no_wiping_frames,
        'hand_bbox':            state.hand_bbox,
        'panel_rect':           state.panel_rect,
        'all_sequences_done':   state.all_sequences_done,
        'panel_count_today':    state.panel_count_today,
    })


@app.route('/api/next_sequence', methods=['POST'])
def next_sequence_manual():
    ok = advance_sequence()
    return jsonify({'success': ok,
                    'current_sequence': state.current_sequence})


@app.route('/api/capture_now', methods=['POST'])
def capture_now():
    # FIX CRASH3: numpy arrays cannot be used with Python "or" —
    # use explicit is-None guards throughout.
    best_frame = state.last_clean_frame
    if best_frame is None: best_frame = state.orig_frame
    if best_frame is None:
        return jsonify({'success': False, 'error': 'No frame available'}), 400
    _pr = state.last_clean_panel_rect
    if _pr is None: _pr = state.panel_rect
    _pc = state.last_clean_contour
    if _pc is None: _pc = state.panel_contour
    cap_data = {
        'frame':         best_frame,
        'panel_rect':    _pr,
        'panel_contour': _pc,
        'serial_det':    state.last_clean_serial_det or state.serial_det,
    }
    saved = capture_sequence_images(cap_data, state.current_sequence, state.serial_number)
    if saved:
        state.sequence_captured[state.current_sequence] = True
        return jsonify({'success': True,
                        'sequence': state.current_sequence})
    return jsonify({'success': False, 'error': 'Capture failed'}), 500


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*60)
    print("[OK] PANEL VISION INSPECTION SYSTEM")
    print("    Vidana Consulting Pvt Ltd")
    print("="*60)
    be = state.backend.upper()
    print(f"[INFO] Backend  : {be}")
    if state.backend in ("cpu", "gpu"):
        print(f"[INFO] Model    : {PT_MODEL_PATH}")
    else:
        print("[WARN] No model — hand detection disabled")
    print(f"📁  Storage  : {os.path.abspath(BASE_STORAGE)}")
    print(f"🌐  Server   : http://0.0.0.0:5000")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

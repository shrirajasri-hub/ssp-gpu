"""
panel_logger.py — Panel Vision Inspection System
=================================================
Structured, timestamped logging for the complete panel lifecycle.

Two log files per day:
  panel_data/YYYY-MM-DD/app_log_YYYY-MM-DD.txt
      → daily log: app start/stop, camera health, one-line per panel
  panel_data/YYYY-MM-DD/SSP-SEQ_HHMMSS/panel_log.txt
      → per-panel: full lifecycle, every image, OCR attempt, failure reasons

Usage (app_vision.py):
    from panel_logger import get_logger
    log = get_logger()
    log.app_start(cam1_url, cam2_url)
    log.panel_start(panel_id, folder)
    log.seq_detected(1, conf=0.92)
    log.image_saved("SEQ1_Full", path)
    log.serial_confirmed("010625123A", elapsed=2.6)
    log.panel_end("010625123A", result="SUCCESS")

Usage (camera2_ocr.py):
    from panel_logger import get_logger
    log = get_logger()
    log.cam2_yolo_detection(conf, crop_w, crop_h, sharpness)
    log.ocr_called(crop_w, crop_h)
    log.ocr_result(raw_text, score, corrected)
"""

import os
import threading
from datetime import datetime

# ── Module-level singleton ────────────────────────────────────────────────────
_instance = None
_instance_lock = threading.Lock()


def get_logger():
    """Return the module-level PanelLogger singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PanelLogger()
    return _instance


def _shift_name(dt=None):
    """Return shift name based on hour."""
    h = (dt or datetime.now()).hour
    if 6 <= h < 14:
        return "Shift-1 (Morning 06:00–14:00)"
    elif 14 <= h < 22:
        return "Shift-2 (Afternoon 14:00–22:00)"
    else:
        return "Shift-3 (Night 22:00–06:00)"


def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _ts_full():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class PanelLogger:
    """
    Thread-safe logger for the complete panel inspection lifecycle.
    Writes to two files:
      1. app_log_{date}.txt  — daily summary
      2. panel_log.txt       — per-panel detail (inside each panel folder)
    """

    def __init__(self):
        self._lock          = threading.Lock()
        self._base_dir      = None       # panel_data/ root (set in init)
        self._daily_path    = None       # app_log_YYYY-MM-DD.txt
        self._panel_path    = None       # current panel_log.txt
        self._panel_folder  = None       # current panel folder path
        self._panel_start   = None       # datetime of panel start
        self._panel_id      = None       # e.g. "105210"
        self._panel_serial  = None       # confirmed serial or None
        self._panel_number  = 0          # counter for this session
        self._app_start     = None       # datetime of app start
        self._cam2_frame_count   = 0
        self._cam2_last_frame_ts = None
        self._ocr_start_ts  = None
        self._seq1_start    = None
        self._seq2_start    = None
        self._seq3_start    = None

    # ── INTERNAL ─────────────────────────────────────────────────────────────

    def _setup_daily(self, base_dir):
        """Ensure daily log file path is current. Called at init and on every write."""
        self._base_dir = base_dir
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_folder = os.path.join(base_dir, date_str)
        os.makedirs(date_folder, exist_ok=True)
        self._daily_path = os.path.join(date_folder,
                                        f"app_log_{date_str}.txt")
        self._current_log_date = date_str

    def _daily(self, msg: str):
        """Write a line to the daily log. Auto-rotates at midnight."""
        if not self._daily_path:
            return
        # Midnight rotation: if date has changed, open new file
        today = datetime.now().strftime("%Y-%m-%d")
        if today != getattr(self, '_current_log_date', today):
            self._setup_daily(self._base_dir)
            self._daily_raw(
                f"\n{'═'*70}\n"
                f"DATE CHANGED → {today}  (log rotated at midnight)\n"
                f"{'═'*70}\n"
            )
        line = f"[{_ts()}] {msg}\n"
        try:
            with open(self._daily_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def _panel(self, msg: str):
        """Write a line to the current panel log."""
        if not self._panel_path:
            return
        line = f"[{_ts()}] {msg}\n"
        try:
            with open(self._panel_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def _both(self, msg: str):
        """Write to both logs."""
        self._daily(msg)
        self._panel(msg)

    def _panel_raw(self, text: str):
        """Write raw text (no timestamp prefix) to panel log."""
        if not self._panel_path:
            return
        try:
            with open(self._panel_path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    def _daily_raw(self, text: str):
        if not self._daily_path:
            return
        try:
            with open(self._daily_path, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass

    def set_panel_folder(self, folder: str):
        """Point panel log to the given panel folder."""
        with self._lock:
            self._panel_folder = folder
            if folder:
                self._panel_path = os.path.join(folder, "panel_log.txt")

    # ── APPLICATION LIFECYCLE ────────────────────────────────────────────────

    def init(self, base_dir: str):
        """Call once at app startup with BASE_STORAGE path."""
        with self._lock:
            self._setup_daily(base_dir)

    def app_start(self, cam1_url: str = "", cam2_url: str = ""):
        """Log application startup."""
        self._app_start = datetime.now()
        shift = _shift_name(self._app_start)
        with self._lock:
            self._daily_raw(
                f"\n{'═'*70}\n"
                f"APPLICATION STARTED   {_ts_full()}\n"
                f"Shift : {shift}\n"
                f"CAM1  : {cam1_url}\n"
                f"CAM2  : {cam2_url}\n"
                f"{'═'*70}\n"
            )

    def app_stop(self, panels_total: int = 0,
                 panels_success: int = 0, panels_failed: int = 0):
        elapsed = ""
        if self._app_start:
            d = datetime.now() - self._app_start
            h, r = divmod(int(d.total_seconds()), 3600)
            m, s = divmod(r, 60)
            elapsed = f"  uptime={h}h{m:02d}m{s:02d}s"
        with self._lock:
            self._daily_raw(
                f"\n{'═'*70}\n"
                f"APPLICATION STOPPED   {_ts_full()}{elapsed}\n"
                f"Panels : {panels_total} total  "
                f"✅ {panels_success} success  "
                f"❌ {panels_failed} failed\n"
                f"{'═'*70}\n\n"
            )

    def model_loaded(self, name: str, path: str,
                     size_mb: float = 0, device: str = ""):
        msg = (f"MODEL LOADED        {name}"
               f"  path={path}"
               f"  size={size_mb:.1f}MB"
               f"  device={device}")
        with self._lock:
            self._daily(msg)

    def model_failed(self, name: str, reason: str):
        msg = f"MODEL FAILED    ❌  {name}  reason={reason}"
        with self._lock:
            self._daily(msg)

    # ── CAMERA HEALTH ────────────────────────────────────────────────────────

    def camera_connected(self, cam_id: int, url: str):
        msg = f"CAM{cam_id} CONNECTED       {url}"
        with self._lock:
            self._daily(msg)

    def camera_drop(self, cam_id: int, fps_before: float = 0):
        msg = (f"CAM{cam_id} STREAM DROP  ⚠️  "
               f"frames stopped  (was {fps_before:.0f}fps)")
        with self._lock:
            self._daily(msg)
            self._panel(msg)

    def camera_reconnecting(self, cam_id: int, attempt: int):
        msg = f"CAM{cam_id} RECONNECTING   attempt {attempt}"
        with self._lock:
            self._daily(msg)

    def camera_reconnected(self, cam_id: int, fps: float = 0):
        msg = f"CAM{cam_id} RECONNECTED  ✅  frames resumed ({fps:.0f}fps)"
        with self._lock:
            self._daily(msg)
            self._panel(msg)

    def cam2_no_frames(self, elapsed_s: float):
        msg = (f"CAM2 NO FRAMES      ❌  No frames in {elapsed_s:.0f}s  "
               f"ROOT CAUSE: Camera2 RTSP not connected or stream dropped")
        with self._lock:
            self._daily(msg)
            self._panel(msg)

    def cam2_frame_stats(self, frame_count: int, fps: float,
                         sharpness_avg: float):
        msg = (f"CAM2 FRAMES         count={frame_count}"
               f"  fps={fps:.1f}"
               f"  sharpness_avg={sharpness_avg:.0f}")
        with self._lock:
            self._panel(msg)

    # ── PANEL LIFECYCLE ──────────────────────────────────────────────────────

    def panel_start(self, panel_id: str, folder: str):
        """Call when panel folder is created and session begins."""
        self._panel_start  = datetime.now()
        self._panel_id     = panel_id
        self._panel_serial = None
        self._ocr_start_ts = None
        self._seq1_start   = None
        self._seq2_start   = None
        self._seq3_start   = None
        with self._lock:
            self._panel_number += 1
            self.set_panel_folder(folder)

            # Panel heading in panel_log.txt
            shift = _shift_name(self._panel_start)
            self._panel_raw(
                f"{'═'*70}\n"
                f"PANEL #{self._panel_number:04d}  |  "
                f"Serial: (scanning…)  |  "
                f"{self._panel_start.strftime('%Y-%m-%d  %H:%M:%S')}\n"
                f"Panel ID  : {panel_id}\n"
                f"Shift     : {shift}\n"
                f"Folder    : {folder}\n"
                f"{'═'*70}\n\n"
            )

            # One-line entry in daily log
            self._daily(
                f"--- PANEL #{self._panel_number:04d} STARTED ---  "
                f"panel_id={panel_id}  shift={shift.split()[0]}"
            )

    def panel_serial_known(self, serial: str):
        """Update heading once serial is confirmed."""
        self._panel_serial = serial
        # Append note to panel log
        with self._lock:
            self._panel(f"SERIAL KNOWN        {serial}  (heading updated below)")

    def panel_end(self, serial: str, result: str,
                  images_saved: int = 0, pdf_path: str = "",
                  failure_reason: str = "",
                  seq_durations: dict = None,
                  capture_statuses: dict = None):
        """Call at panel completion (success or failure)."""
        end_time = datetime.now()
        elapsed  = ""
        lead_s   = 0.0
        if self._panel_start:
            d      = end_time - self._panel_start
            lead_s = d.total_seconds()
            m, s   = divmod(int(lead_s), 60)
            elapsed = f"{m}m {s:02d}s"

        ok = result.upper() in ("SUCCESS", "✅", "COMPLETE")
        icon = "✅" if ok else "❌"

        seq_durations = seq_durations or {}
        capture_statuses = capture_statuses or {}

        def _fmt_time(s):
            return f"{s:.1f}s" if s is not None else "0.0s"

        with self._lock:
            # Footer in panel_log.txt
            self._panel_raw(
                f"\n{'═'*70}\n"
                f"PANEL {'COMPLETE' if ok else 'FAILED  ❌'}  "
                f"serial={serial or 'UNKNOWN'}\n"
                f"Started  : {self._panel_start.strftime('%H:%M:%S') if self._panel_start else '?'}\n"
                f"Finished : {end_time.strftime('%H:%M:%S')}\n"
                f"Panel Duration: {elapsed}  ({lead_s:.1f}s)\n"
                f"SEQ1 Duration : {_fmt_time(seq_durations.get(1))}\n"
                f"SEQ2 Duration : {_fmt_time(seq_durations.get(2))}\n"
                f"SEQ3 Duration : {_fmt_time(seq_durations.get(3))}\n"
                f"Capture Status: SEQ1={capture_statuses.get(1, 'MISSING')}  "
                f"SEQ2={capture_statuses.get(2, 'MISSING')}  "
                f"SEQ3={capture_statuses.get(3, 'MISSING')}\n"
                f"Images   : {images_saved}\n"
                f"PDF      : {os.path.basename(pdf_path) if pdf_path else 'not generated'}\n"
            )
            if not ok and failure_reason:
                self._panel_raw(
                    f"FAILURE REASON: {failure_reason}\n"
                )
            self._panel_raw(f"{'═'*70}\n\n")

            # Summary in daily log
            self._daily(
                f"--- PANEL #{self._panel_number:04d} "
                f"{'COMPLETED' if ok else 'FAILED  ❌'} --- "
                f"serial={serial or 'UNKNOWN'}  "
                f"lead_time={elapsed}  "
                f"images={images_saved}  "
                f"{icon}"
                + (f"  reason={failure_reason}" if not ok and failure_reason else "")
            )

    # ── SEQUENCE EVENTS ──────────────────────────────────────────────────────

    def seq_detected(self, seq_num: int, conf: float,
                     stable_frames: int = 0):
        if seq_num == 1:
            self._seq1_start = datetime.now()
        elif seq_num == 2:
            self._seq2_start = datetime.now()
        elif seq_num == 3:
            self._seq3_start = datetime.now()
        msg = (f"SEQ{seq_num} DETECTED       conf={conf:.3f}"
               f"  stable_frames={stable_frames}")
        with self._lock:
            self._panel(f"── SEQ{seq_num} {'─'*53}")
            self._panel(msg)

    def seq_wipe_start(self, wiping_frame: int):
        msg = f"SEQ1 WIPE START     wiping_frame={wiping_frame}"
        with self._lock:
            self._panel(msg)

    def seq_portrait(self, seq_num: int):
        msg = f"SEQ{seq_num} PORTRAIT      panel rotated — serial face visible"
        with self._lock:
            self._panel(msg)

    def seq_landscape(self, seq_num: int):
        msg = f"SEQ{seq_num} LANDSCAPE     flip panel to landscape for capture"
        with self._lock:
            self._panel(msg)

    # ── IMAGE STORAGE ────────────────────────────────────────────────────────

    def image_saved(self, label: str, file_path: str, extra: str = ""):
        msg = (f"IMAGE SAVED     ✅  {label:<22} "
               f"file={os.path.basename(file_path)}"
               + (f"  {extra}" if extra else ""))
        with self._lock:
            self._panel(msg)

    def image_failed(self, label: str, reason: str):
        msg = (f"IMAGE FAILED    ❌  {label:<22} "
               f"ROOT CAUSE: {reason}")
        with self._lock:
            self._panel(msg)
            self._daily(f"IMAGE FAILED  {label}  {reason}")

    def images_summary(self, seq_num: int, files: list):
        """Log all files saved for a sequence."""
        with self._lock:
            for f in files:
                self._panel(f"  SAVED           {os.path.basename(f)}")

    def capture_event(self, seq_name: str, status: str):
        msg = f"CAPTURE EVENT   ✅  {seq_name:<10}  status={status}"
        with self._lock:
            self._panel(msg)

    def missed_detection_event(self, seq_name: str, reason: str):
        msg = f"MISSED DETECT   ⚠️  {seq_name:<10}  reason={reason}"
        with self._lock:
            self._panel(msg)

    def serial_trace(self, trace_type: str, frame_idx: int, conf: float):
        msg = f"SERIAL TRACE    🔍  {trace_type:<10} frame={frame_idx}  conf={conf:.2f}"
        with self._lock:
            self._panel(msg)

    # ── OCR / SERIAL ─────────────────────────────────────────────────────────

    def ocr_triggered(self, wiping_frame: int, trigger: str = ""):
        self._ocr_start_ts = datetime.now()
        msg = (f"OCR TRIGGERED   ✅  wiping_frame={wiping_frame}"
               + (f"  via={trigger}" if trigger else ""))
        with self._lock:
            self._panel(f"── OCR {'─'*55}")
            self._panel(msg)

    def cam2_yolo_detection(self, conf: float, crop_w: int,
                            crop_h: int, sharpness: float, queue_size: int = 0):
        msg = (f"YOLO DETECTION      conf={conf:.3f}"
               f"  crop={crop_w}×{crop_h}"
               f"  sharp={sharpness:.0f}"
               f"  queue={queue_size}")
        with self._lock:
            self._panel(msg)

    def cam2_crop_rejected(self, reason: str, sharpness: float = 0):
        msg = (f"CROP REJECTED   ❌  sharpness={sharpness:.1f}  "
               f"ROOT CAUSE: {reason}")
        with self._lock:
            self._panel(msg)

    def cam2_yolo_no_detection(self, frames_checked: int):
        msg = (f"YOLO NO DETECT  ❌  {frames_checked} frames checked  "
               f"ROOT CAUSE: serial_number class not detected — "
               f"wrong panel face or low model confidence")
        with self._lock:
            self._panel(msg)
            if frames_checked % 60 == 0:
                self._daily(f"YOLO NO DETECT  {frames_checked} frames  "
                            f"panel_id={self._panel_id}")

    def ocr_called(self, crop_w: int, crop_h: int):
        msg = f"PADDLEOCR CALLED    input={crop_w*2}×{crop_h*2}  (2× resize)"
        with self._lock:
            self._panel(msg)

    def ocr_result(self, raw: str, score: float, corrected):
        if corrected:
            msg = (f"OCR RESULT      ✅  raw={repr(raw):<22}"
                   f"  score={score:.3f}  → {corrected}")
        else:
            msg = (f"OCR RESULT      ❌  raw={repr(raw):<22}"
                   f"  score={score:.3f}  INVALID FORMAT")
        with self._lock:
            self._panel(msg)

    def ocr_no_text(self):
        msg = ("OCR RESULT      ❌  (no text returned)  "
               "ROOT CAUSE: crop too blurry / wrong region / PaddleOCR model issue")
        with self._lock:
            self._panel(msg)

    def ocr_paddleocr_loading(self, elapsed_s: float):
        msg = (f"PADDLEOCR       ⏳  still loading ({elapsed_s:.0f}s elapsed)  "
               f"ROOT CAUSE: PaddleOCR takes 30–120s on first run")
        with self._lock:
            self._panel(msg)

    def serial_vote(self, day: str, code: str, letter: str,
                    d_cnt: int, c_cnt: int, l_cnt: int, cv: int):
        msg = (f"SERIAL VOTE         "
               f"day='{day}'({d_cnt}/{cv})  "
               f"code='{code}'({c_cnt}/{cv})  "
               f"letter='{letter}'({l_cnt}/{cv})")
        with self._lock:
            self._panel(msg)

    def serial_frozen(self, position: str, value: str):
        msg = f"POSITION FROZEN ⭐  {position}='{value}'"
        with self._lock:
            self._panel(msg)

    def serial_confirmed(self, serial: str, elapsed_s: float = 0):
        self._panel_serial = serial
        msg = (f"SERIAL CONFIRMED ✅  {serial}  "
               f"time_to_confirm={elapsed_s:.1f}s")
        with self._lock:
            self._panel(f"── SERIAL CONFIRMED {'─'*42}")
            self._panel(msg)
            self._daily(f"SERIAL CONFIRMED    {serial}  "
                        f"panel_id={self._panel_id}  t={elapsed_s:.1f}s")

    def serial_failed(self, reason: str, frames_tried: int = 0,
                      crops_tried: int = 0):
        msg = (f"SERIAL FAILED   ❌  "
               f"frames={frames_tried}  crops={crops_tried}\n"
               f"                    ROOT CAUSE: {reason}")
        with self._lock:
            self._panel(f"── SERIAL FAILED {'─'*45}")
            self._panel(msg)
            self._daily(f"SERIAL FAILED       panel_id={self._panel_id}  "
                        f"reason={reason}")

    # ── PDF ──────────────────────────────────────────────────────────────────

    def pdf_start(self, serial: str):
        msg = f"PDF GENERATING      serial={serial}"
        with self._lock:
            self._panel(f"── PDF {'─'*55}")
            self._panel(msg)

    def pdf_complete(self, path: str, pages: int):
        msg = (f"PDF COMPLETE    ✅  file={os.path.basename(path)}"
               f"  pages={pages}")
        with self._lock:
            self._panel(msg)
            self._daily(f"PDF COMPLETE        {os.path.basename(path)}")

    def pdf_failed(self, reason: str):
        msg = f"PDF FAILED      ❌  ROOT CAUSE: {reason}"
        with self._lock:
            self._panel(msg)
            self._daily(f"PDF FAILED          panel_id={self._panel_id}  {reason}")

    # ── FOLDER ───────────────────────────────────────────────────────────────

    def folder_created(self, path: str, panel_id: str):
        msg = (f"FOLDER CREATED  ✅  {path}"
               f"  panel_id={panel_id}")
        with self._lock:
            self._panel(msg)

    def folder_renamed(self, old: str, new: str):
        msg = (f"FOLDER RENAMED  ✅  "
               f"{os.path.basename(old)} → {os.path.basename(new)}")
        with self._lock:
            self._panel(msg)
            self._daily(msg)

    def cam2_folder_set(self, path: str):
        msg = f"CAM2 FOLDER SET ✅  {path}"
        with self._lock:
            self._panel(msg)

    # ── GENERIC ──────────────────────────────────────────────────────────────

    def info(self, msg: str):
        with self._lock:
            self._panel(msg)

    def warn(self, msg: str):
        with self._lock:
            self._panel(f"⚠️  {msg}")
            self._daily(f"WARNING  {msg}")

    def error(self, msg: str):
        with self._lock:
            self._panel(f"❌  {msg}")
            self._daily(f"ERROR    {msg}")

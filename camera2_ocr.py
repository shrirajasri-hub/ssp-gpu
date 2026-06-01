# -*- coding: utf-8 -*-
"""
camera2_ocr.py  â  Panel Vision  |  Vidana Consulting Pvt Ltd
==============================================================
Camera-2 serial number pipeline.

Flow
ââââ
 1. serial.pt runs on EVERY incoming Camera-2 frame (always-on).
 2. When serial class is detected â green bbox drawn for UI via
    get_annotated_frame().  No detection â clean frame, no overlay.
 3. Capture slots (Ã3) filled only when YOLO returns a real detection
    and the crop passes minimum sharpness.  NO fixed-ROI fallback.
 4. 3 unannotated raw frames + crops saved to panel folder.
 5. Full preprocessing pipeline (CLAHE, blackhat, tophat, bilateralâ¦)
    YOLO crop (padded 2x) -> PaddleOCR predict -> voting -> serial.
 6. Confirmed serial written to serial_ocr_result.txt and PDF.

GPU/CPU backend
âââââââââââââââ
 serial.pt uses PyTorch on CUDA GPU (automatic fallback to CPU).
 app_vision.py also uses GPU/CPU for best.pt.
 No Hailo hardware required.
"""

print("\n" + "="*62)
print(">>> [SYSTEM] Camera2OCR v3.2 (ROUND_ROBIN + YOLO-only) <<<")
print("="*62 + "\n")

import cv2
import os
import re
import threading
import time
import subprocess
import platform
import numpy as np
import select
from datetime import datetime
from collections import Counter
import queue as _queue_mod

# ââ Serial number correction maps (from video_test_03_cv2.py) ââ
import string as _string

_REPL_DIGITS = {
    '/':'7', '|':'1', 'I':'1', 'i':'1',
    ' ':'_', 'J':'1', ':':'_', '*':'_', ';':'_',
}
_REPL_CHAR = {'4':'A', '^':'A', 'a':'A', '8':'B'}

def _apply_corrections(s: str):
    """Exact port of reference validate_serial_strict() + correct_serial()."""
    if len(s) not in (6, 10):
        return None

    # ââ Extract positions based on format ââââââââââââââââââââ
    if len(s) == 6:
        # DD + 3digits + Letter (short format â system adds MMYY)
        day_s, code_s, letter_s = s[0:2], s[2:5], s[5]
    else:  # 10-char
        # DD + MMYY (skip) + 3digits + Letter
        day_s, code_s, letter_s = s[0:2], s[6:9], s[9]

    # ââ Digit corrections for day and code âââââââââââââââââââ
    def fix_digits(t):
        return (t.replace('l','1').replace('L','1')
                 .replace('i','1').replace('I','1')
                 .replace('O','0').replace('o','0')
                 .replace('/','7').replace('|','1'))

    day_s  = fix_digits(day_s.upper())
    code_s = fix_digits(code_s.upper())

    # ââ Validate day (01-31) ââââââââââââââââââââââââââââââââââ
    try:
        day_int = int(day_s)
        if not (1 <= day_int <= 31): return None
    except ValueError: return None

    # ââ Validate code (3 digits) ââââââââââââââââââââââââââââââ
    if not (code_s.isdigit() and len(code_s) == 3): return None

    # ââ Letter corrections (exact from reference) âââââââââââââ
    letter_s = letter_s.upper()
    if   letter_s == '4':            letter_s = 'A'
    elif letter_s in ('3', '8'):     letter_s = 'B'
    elif letter_s == '^':            letter_s = 'A'
    elif letter_s == 'O':            letter_s = 'A'
    elif letter_s == 'a':            letter_s = 'A'
    if letter_s not in ('A','B','C','D'): return None

    # ââ Build final 10-char serial with system month/year âââââ
    from datetime import datetime as _dt
    _now = _dt.now()
    mm = f'{_now.month:02d}'
    yy = f'{_now.year % 100:02d}'
    return day_s + mm + yy + code_s + letter_s

def _correct_serial(raw: str):
    """
    Clean OCR noise then extract valid 9digit+1letter serial.
    Handles: embedded spaces/punctuation, 1-2 extra chars at start/end.
    Ported from video_test_03_cv2.py â proven working on Pi.
    """
    s = raw.strip().strip("'`\"")
    for k, v in _REPL_DIGITS.items():
        s = s.replace(k, v)
    s = re.sub(r'[^A-Za-z0-9]', '', s)
    if not (9 <= len(s) <= 12):
        return None
    if len(s) == 10:
        return _apply_corrections(s)
    if len(s) == 11:
        return _apply_corrections(s[1:]) or _apply_corrections(s[:-1])
    if len(s) == 12:
        return (_apply_corrections(s[2:]) or _apply_corrections(s[:-2]) or
                _apply_corrections(s[1:-1]))
    return None

# ââ OCR engine: PaddleOCR
_TESSERACT_OK = False
try:
    # (easyocr removed — using PaddleOCR)
    _PADDLE_OK = True
    # (PaddleOCR loaded above)
except ImportError:
    _PADDLE_OK = False

# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  SHARPNESS SCORING
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def sharpness_score(img: np.ndarray) -> float:
    """Laplacian variance â higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  REMOVED: SerialHailoDetector (Hailo HEF runtime removed).
#  GPU build uses SerialYOLODetector (PyTorch/Ultralytics) below.
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class SerialHailoDetector:
    """
    STUB â Hailo runtime removed. This class is never instantiated.
    Camera2OCR uses SerialYOLODetector (PyTorch GPU/CPU) exclusively.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Hailo runtime removed â use SerialYOLODetector (GPU/CPU)"
        )


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  PREPROCESSING  â  grey panel + engraved text
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def preprocess_engraved_metal(roi_bgr: np.ndarray,
                               save_dir: str = None) -> list:
    """
    Reduced high-value variants for engraved text on grey metal to save CPU.
    PaddleOCR (no preprocessing variants needed): continuous greyscale (CNN-friendly).
    Upscaled minimally (â¥2Ã) to balance accuracy and CPU cost.
    """
    gray = (cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            if roi_bgr.ndim == 3 else roi_bgr.copy())
    h, w  = gray.shape
    # Upscale moderately (min 2Ã). Keeps OCR-friendly resolution
    # while avoiding very large upscales on Pi-class CPUs.
    scale = max(2.0, min(1280 / max(w, 1), 3.0))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cg    = clahe.apply(gray)
    v     = []

    def up(img):
        return cv2.resize(img, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # 1 â CLAHE + adaptive threshold
    adapt = cv2.adaptiveThreshold(
        cg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 4)
    adapt = cv2.morphologyEx(
        adapt, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
    v1 = up(adapt)
    v += [("clahe_adapt", v1)]

    # 2 â Blackhat: dark engravings
    kb = cv2.getStructuringElement(cv2.MORPH_RECT, (25,7))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kb)
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    _,bh = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    v.append(("blackhat", up(bh)))

    # 3 â Tophat: light ridges
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kb)
    th = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX)
    _,th = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    v.append(("tophat", up(th)))

    # 4 â Sharpening + CLAHE
    blur = cv2.GaussianBlur(gray, (0,0), 2.0)
    shp  = np.clip(cv2.addWeighted(gray,2.5,blur,-1.5,0),0,255).astype(np.uint8)
    shp  = clahe.apply(shp)
    v4   = up(shp)
    v += [("sharp_clahe", v4), ("sharp_clahe_inv", cv2.bitwise_not(v4))]

    # 5 â Morphological gradient
    kg  = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    grd = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kg)
    grd = clahe.apply(grd)
    _,grd = cv2.threshold(grd,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    v.append(("gradient", up(grd)))

    # 6 â Histogram equalization + 2D enhancement filter  [PaddleOCR]
    heq = cv2.equalizeHist(gray)
    k2d = np.array([[-1,-1,-1,-1,-1],
                    [-1, 2, 2, 2,-1],
                    [-1, 2, 8, 2,-1],
                    [-1, 2, 2, 2,-1],
                    [-1,-1,-1,-1,-1]], dtype=np.float32) / 8.0
    flt = np.clip(cv2.filter2D(heq,-1,k2d),0,255).astype(np.uint8)
    v.append(("heq_2d_filter", up(flt)))

    # --- Additional variants: Otsu binarization on upscaled image
    up_orig = up(gray)
    # simple Otsu on upscaled original
    _, otsu = cv2.threshold(up_orig, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v.append(("orig_otsu", otsu))

    # 2D filter then Otsu
    filt2d_up = cv2.filter2D(up_orig, -1, k2d)
    _, otsu2 = cv2.threshold(filt2d_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v.append(("2d_otsu", otsu2))

    # 7 â Bilateral filter + CLAHE  [PaddleOCR]
    bil = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    v.append(("bilateral_clahe", up(clahe.apply(bil))))

    # GPU: use all variants; Pi/CPU: 3 fastest
    _ALL = ("clahe_adapt","blackhat","tophat","sharp_clahe",
            "sharp_clahe_inv","gradient","heq_2d_filter","2d_otsu","bilateral_clahe")
    _CPU = ("clahe_adapt","heq_2d_filter","2d_otsu")
    try:
        import torch as _t; _keep = _ALL if _t.cuda.is_available() else _CPU
    except Exception:
        _keep = _CPU
    v = [(lbl, img) for (lbl, img) in v if lbl in _keep]

    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            for label, img in v:
                cv2.imwrite(os.path.join(save_dir, f"pp_{label}.jpg"), img)
        except Exception:
            pass
    return v


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  HWACCEL DETECTION â detects NVIDIA (Z440) or ARM (Pi) at startup
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def _detect_ffmpeg_hwaccel():
    """
    Returns 'nvidia' | 'v4l2m2m' | 'software'.
    - nvidia   â NVIDIA GPU (Z440/workstation): use h264_cuvid
    - v4l2m2m  â Raspberry Pi ARM:             use v4l2m2m
    - software â everything else:              pure libavcodec
    """
    try:
        r   = subprocess.run(['ffmpeg', '-hwaccels'],
                             capture_output=True, text=True, timeout=4)
        out = r.stdout.lower()
        is_arm = ('arm' in platform.machine().lower() or
                  'aarch' in platform.machine().lower())
        has_cuda  = 'cuda'     in out or 'cuvid' in out
        has_v4l2  = 'v4l2m2m' in out

        if has_cuda and not is_arm:
            print('[HWACCEL] NVIDIA GPU detected â h264_cuvid â')
            return 'nvidia'
        if is_arm and has_v4l2:
            print('[HWACCEL] ARM Pi detected â v4l2m2m (software-safe flags)')
            return 'v4l2m2m'
    except Exception as e:
        print(f'[HWACCEL] probe failed ({e})')
    print('[HWACCEL] Software decode (libavcodec)')
    return 'software'

_HW_TYPE = _detect_ffmpeg_hwaccel()


def _ffmpeg_hw_prefix():
    """
    Returns list of FFmpeg flags to insert BEFORE -i.
    NVIDIA â h264_cuvid (GPU decode, CPU-memory output â vf works fine).
    ARM    â software decode (v4l2m2m caused NV12 colour errors on Pi).
    Other  â software decode.
    """
    if _HW_TYPE == 'nvidia':
        # h264_cuvid: GPU H264 decode, output in regular YUV420P system RAM
        # so downstream software vf filters (scale, format=bgr24) work correctly.
        return ['-hwaccel', 'cuvid', '-c:v', 'h264_cuvid']
    return []   # software decode for ARM and all other platforms


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  SERIAL YOLO DETECTOR  â replaces SerialHailoDetector on Z440 GPU
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class SerialYOLODetector:
    """
    Drop-in replacement for SerialHailoDetector.
    Uses best.pt (ultralytics YOLOv8) on CUDA GPU (Z440/workstation)
    or CPU as fallback.  Filters detections to 'serial_number' class only.

    Interface is identical to SerialHailoDetector:
      detector.detect(frame_bgr) â [(x1,y1,x2,y2,conf)]
      detector.annotate(frame, det, label) â annotated frame
    """

    CROP_PAD_PX    = 30
    # Some YOLO exports may label the serial class as 'serial' or
    # 'serial_number'. Accept both so annotations are not dropped.
    VALID_SERIAL_CLASSES = {'serial', 'serial_number'}
    CONF_THRESHOLD = 0.30

    def __init__(self, pt_path: str):
        from ultralytics import YOLO
        import torch

        self._device = 'cuda:0' if self._cuda_usable() else 'cpu'
        self.model   = YOLO(pt_path)

        try:
            self.model.predict(
                source=__import__('numpy').zeros((640, 640, 3), dtype='uint8'),
                device=self._device, verbose=False)
        except Exception as e:
            if self._device != 'cpu':
                print(f"[CAM2-YOLO] CUDA warmup failed: {e}. Falling back to CPU.")
                self._device = 'cpu'
                self.model.predict(
                    source=__import__('numpy').zeros((640, 640, 3), dtype='uint8'),
                    device='cpu', verbose=False)
            else:
                raise

        print(f'[CAM2-YOLO] SerialYOLODetector ready '
              f'device={self._device}  model={pt_path}  â')

    @staticmethod
    def _cuda_usable():
        import torch
        if not torch.cuda.is_available():
            return False
        try:
            torch.zeros(1, device='cuda:0')
            return True
        except Exception as e:
            print(f'[CAM2-YOLO] CUDA available but unusable: {e}')
            return False

    def detect(self, frame_bgr):
        """
        Run YOLOv8 inference, return serial_number detections only.
        Returns list of (x1, y1, x2, y2, conf) in frame pixel coords.
        """
        try:
            results = self.model.predict(
                source=frame_bgr,
                device=self._device,
                conf=self.CONF_THRESHOLD,
                verbose=False)
            dets = []
            if results and len(results[0].boxes) > 0:
                names = results[0].names  # {id: name}
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    cls_name = names.get(cls_id, '')
                    # DEBUG: log the first detection to see actual class names
                    if not hasattr(self, '_logged_class_names'):
                        print(f'[CAM2-YOLO] First detection: cls_id={cls_id}, cls_name="{cls_name}"')
                        print(f'[CAM2-YOLO] All classes map: {names}')
                        self._logged_class_names = True
                    
                    if cls_name not in self.VALID_SERIAL_CLASSES and cls_id != 4:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in
                                       box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    dets.append((x1, y1, x2, y2, conf))
                    if not hasattr(self, '_serial_det_logged'):
                        print(f'[CAM2-YOLO] â Serial detection accepted: "{cls_name}" (conf={conf:.2f})')
                        self._serial_det_logged = True
            return dets
        except Exception as e:
            print(f'[CAM2-YOLO] detect() error: {e}')
            return []

    def annotate(self, frame_bgr, det, label='serial_number'):
        """Draw green bbox on frame. det = (x1,y1,x2,y2,conf)."""
        import cv2
        out = frame_bgr.copy()
        if det is not None:
            x1, y1, x2, y2, conf = det
            
            # Apply visual padding to the UI annotation
            ih, iw = out.shape[:2]
            p = self.CROP_PAD_PX
            x1 = max(0, x1 - p)
            y1 = max(0, y1 - p)
            x2 = min(iw, x2 + p)
            y2 = min(ih, y2 + p)
            
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 100, 0), 4)
            cv2.putText(out, f'{label} {conf:.2f}',
                        (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 100, 0), 2, cv2.LINE_AA)
        return out

    def close(self):
        pass   # ultralytics handles cleanup


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  FFMPEG CAPTURE
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class _FFmpegCapture:
    PROXY_PORT = 5556

    def __init__(self, url:str, width:int=1280, height:int=960):
        self.url=url; self.width=width; self.height=height
        self.frame_size=width*height*3
        self._lock=threading.Lock(); self._latest=None; self.running=True
        self._proxy_host=self._parse_host(url)
        if ':443' in url:
            self._start_tls_proxy()
            self._connect_url=self._build_proxy_url(url)
        else:
            self._connect_url=url
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _parse_host(url):
        m=re.search(r'@([^:/]+)',url)
        return m.group(1) if m else '192.10.70.192'

    def _start_tls_proxy(self):
        import ssl, socket as _s
        host=self._proxy_host; port=self.PROXY_PORT
        try:
            t=_s.create_connection(('127.0.0.1',port),timeout=0.3); t.close(); return
        except Exception: pass
        def handle(c):
            ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
            try:
                raw=_s.create_connection((host,443),timeout=5)
                cam=ctx.wrap_socket(raw,server_hostname=host)
            except Exception: c.close(); return
            def pipe(a,b):
                try:
                    while True:
                        d=a.recv(8192)
                        if not d: break
                        b.sendall(d)
                except Exception: pass
            threading.Thread(target=pipe,args=(c,cam),daemon=True).start()
            threading.Thread(target=pipe,args=(cam,c),daemon=True).start()
        def srv():
            sv=_s.socket(); sv.setsockopt(_s.SOL_SOCKET,_s.SO_REUSEADDR,1)
            sv.bind(('127.0.0.1',port)); sv.listen(10)
            while self.running:
                try:
                    c,_=sv.accept()
                    threading.Thread(target=handle,args=(c,),daemon=True).start()
                except Exception: break
        threading.Thread(target=srv,daemon=True).start()
        time.sleep(0.5)

    def _build_proxy_url(self,url):
        m=re.match(r'(rtsp://[^@]+@)[^:/]+:?\d*(/.*)',url)
        return f"{m.group(1)}127.0.0.1:{self.PROXY_PORT}{m.group(2)}" if m else url

    def _loop(self):
        # Pure software H264 decode â reliable, correct BGR24 output.
        cmd = ['ffmpeg', '-y', '-loglevel', 'warning',
               '-rtsp_transport', 'tcp',
               '-fflags',  'nobuffer+discardcorrupt',
               '-flags',   'low_delay',
               '-i',       self._connect_url,
               '-vf',      f'scale={self.width}:{self.height},format=bgr24',
               '-vcodec',  'rawvideo',
               '-f',       'rawvideo',
               # 10 fps â fresher than original 5 fps, safe for sequential read
               '-r',       '10',
               '-vsync',   '0',
               '-an', '-sn', '-dn', 'pipe:1']

        _buf = bytearray(self.frame_size)
        _mv  = memoryview(_buf)

        while self.running:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=self.frame_size)

                import threading as _thr
                def _drain_err(p):
                    try:
                        for raw in p.stderr:
                            line = raw.decode('utf-8', errors='ignore').strip()
                            if line:
                                print(f'[CAM2-FFMPEG] {line}')
                    except Exception:
                        pass
                _thr.Thread(target=_drain_err, args=(proc,), daemon=True).start()

                while self.running:
                    rdy, _, _ = select.select([proc.stdout], [], [], 3.0)
                    if not rdy:
                        print('[CAM2] â ï¸  frame read timeout â reconnecting')
                        break

                    # Read exactly one full frame â sequential, never partial
                    # DO NOT use non-blocking drain here: partial reads desync
                    # frame boundaries and corrupt the image (pixels from two
                    # frames get mixed â looks like blur).
                    n = 0
                    try:
                        while n < self.frame_size:
                            got = proc.stdout.readinto(_mv[n:])
                            if not got:
                                break
                            n += got
                    except Exception:
                        break

                    if n == self.frame_size:
                        frm = np.frombuffer(_buf, dtype=np.uint8).reshape(
                            (self.height, self.width, 3)).copy()
                        with self._lock:
                            self._latest = frm  # replace immediately

                proc.kill()
            except Exception as e:
                print(f'[CAM2] _loop error: {e}')
            time.sleep(0.5)

    def read(self):
        with self._lock:
            return ((True,self._latest.copy())
                    if self._latest is not None else (False,None))

    def release(self): self.running=False


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
#  CAMERA-2 OCR
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

class Camera2OCR:
    """
    YOLO-gated serial OCR for Camera-2.

    Rules
    âââââ
    â¢ serial.pt runs on EVERY incoming Camera-2 frame (not just
      during scanning) so the UI annotation is always live.
    â¢ A frame is only eligible for capture / OCR if serial.pt
      returns a bounding box (yolo_detected=True).  Fixed-ROI
      fallback is used for OCR text extraction ONLY.
    â¢ 3 best-sharpness detected frames are saved progressively
      as soon as detections come in â not deferred to OCR time.
    â¢ OCR runs on the 3 saved crops; result voted across all.
    â¢ NO fixed-ROI fallback â serial.pt is the only source of crops.
    """

    # ââ Fixed ROI fallback ââââââââââââââââââââââââââââââââââââââââ
    # Used when serial.pt returns zero detections (stamped/engraved
    # metal serials that the YOLO model was not trained on).
    # Coordinates are pixel positions on the 1280Ã960 Camera-2 feed.
    # Measured from CAM2_Progress_90.jpg grid overlay:
    #   serial stamp "2NYJ2G8Y1A" sits at xâ900-1110, yâ440-505
    #   +30px safety margin on all sides
    SERIAL_ROI       = (870, 410, 1150, 530)   # (x1, y1, x2, y2) on 1280Ã960
    ROI_BOX_COLOR    = (0, 200, 255)            # yellow-orange â distinct from YOLO green

    MIN_SHARPNESS    = 0.0   # Intentionally 0 â engraved/stamped metal serial
                               # numbers have naturally low Laplacian variance;
                               # any non-zero threshold rejects valid frames.
    MIN_SHARPNESS_ROI = 0.0  # Same reason for fixed-ROI fallback path.
    BEST_FRAME_COUNT = 3      # save 3 good frames
    STABLE_COUNT_REQ = 2      # same serial 2Ã = confirmed (was 3)
    OCR_INTERVAL_SEC = 0.4
    SERIAL_REGEX     = re.compile(r'[A-Z0-9]{6,12}')

    # Target timestamps for spread-out captures (seconds from scan start)
    CAPTURE_TARGETS  = [0.0, 1.5, 3.0]

    def __init__(self, camera2_url, on_serial_detected=None,
                 open_cap_fn=None, pt_path=None):
        self.url                  = camera2_url
        self.on_serial_detected   = on_serial_detected
        self.open_cap_fn          = open_cap_fn
        self.running              = True
        self.is_scanning          = False
        self._is_scanning         = False
        self.ocr_done             = False
        self.status               = "Idle"
        self.serial_number        = None
        self.ocr_buffer           = []
        # ââ Position-based voting (exact reference algorithm) âââââ
        self.day_votes            = Counter()
        self.code_votes           = Counter()
        self.letter_votes         = Counter()
        self.day_confirmed        = None
        self.code_confirmed       = None
        self.letter_confirmed     = None
        self.panel_folder         = "."
        self.last_frame           = None
        self.last_success_processed = None
        self.save_burst_frames    = True
        self._frames_saved        = 0
        self._lock                = threading.Lock()
        self._cap                 = None
        self._frame_count         = 0  # total Camera-2 frames received (diagnostics)
        self._yolo_det_count      = 0  # total frames where serial.pt fired a detection

        # app_vision.py compat attributes
        self.partial_serial       = None
        self.cam2_roi_path        = None
        self.cam2_raw_path        = None
        self.intervals_saved      = set()
        self._last_good_ts_global = time.time()
        self._force_reconnect     = False
        self._scan_start_ts       = None
        self.stable_count         = 0
        self.prev_roi             = None
        self._last_yolo_detect_ts  = 0.0  # FIX-11: Track time of last YOLO detection

        # ââ Always-on YOLO state (updated every frame) âââââââââ
        # latest_det: (x1, y1, x2, y2, conf) or None
        self._latest_det          = None
        self._latest_frame        = None   # raw frame for annotation (UI stream)
        self._det_lock            = threading.Lock()
        self._roi_fallback        = False  # True when fixed ROI is active (no YOLO det)

        # ââ Raw frame passthrough â set by main loop BEFORE YOLO â
        # Allows UI stream to start immediately without waiting for
        # YOLO to load or process its first frame.
        self._raw_frame           = None
        self._raw_frame_lock      = threading.Lock()

        # ââ Capture slots: one per CAPTURE_TARGET âââââââââââââ
        # Each slot: None until filled with a detected frame dict
        self._capture_slots       = [None] * self.BEST_FRAME_COUNT
        self._slots_saved         = [False] * self.BEST_FRAME_COUNT

        # OCR buffer
        self._ocr_buffer_frames   = []
        self._best_frames_saved   = False

        # ââ Serial-appeared tracking (2 frames â main folder) âââââ
        self._appeared_frame1_saved = False
        self._appeared_frame1_path  = None
        self._appeared_best_sharp   = 0.0
        self._appeared_best_path    = None
        self._best_full_sharp       = 0.0

        # ââ OCR worker thread (separate from main loop) ââââââââ
        # OCR can take 10-60s on Pi (PaddleOCR x 3 frames).
        # Running it in the main loop would freeze YOLO and UI annotation.
        # Solution: main loop signals this event; OCR worker runs independently.
        self._ocr_event           = threading.Event()
        self._ocr_running         = False   # True while OCR worker is active

        # ââ serial.pt + PaddleOCR load in background âââââââââââââââââââââ
        # Both are heavy (2-8s each on Pi). Loading them synchronously
        # would block the main loop and the Camera-2 stream start.
        # Background threads let Camera 2 stream start immediately.
        self.serial_detector  = None   # set by _init_yolo thread when ready
        self.paddle_reader    = None   # set by _init_paddle thread when ready
        self._yolo_loading     = True   # keep variable name for UI compat

        _pt_path_ref = pt_path

        def _init_yolo():
            """Load PyTorch YOLO model in background.
            Always prints to terminal even on Windows â uses flush=True.
            """
            import sys
            # ââ Step 1: Explicit path check before attempting load âââââââââââââ
            if not _pt_path_ref:
                print('[CAM2] â serial.pt path is None â OCR disabled', flush=True)
                self._yolo_loading = False
                return
            if not os.path.exists(_pt_path_ref):
                print(f'[CAM2] â serial.pt NOT FOUND: {_pt_path_ref}', flush=True)
                print('[CAM2] â â Place serial.pt inside the models/ folder next to app_vision.py', flush=True)
                self._yolo_loading = False
                return
            print(f'[CAM2] ð Loading serial.pt: {_pt_path_ref}', flush=True)
            # ââ Step 2: Load model âââââââââââââââââââââââââââââââââââââââââââââ
            try:
                det = SerialYOLODetector(_pt_path_ref)
                self.serial_detector = det
                print(f'[CAM2] â Serial YOLO loaded â device={det._device}', flush=True)
                print(f'[CAM2] â serial.pt ready â annotations will appear on Camera 2', flush=True)
            except Exception as e:
                print(f'[CAM2] â serial.pt FAILED to load: {e}', flush=True)
                import traceback; traceback.print_exc()
                print('[CAM2] â OCR disabled â check ultralytics is installed: pip install ultralytics', flush=True)
            self._yolo_loading = False

        # PaddleOCR init flags — worker polls instead of blocking per crop
        self._paddle_init_done   = False
        self._paddle_init_failed = False

        def _init_paddle():
            """Load PaddleOCR in background thread.
            Tries GPU first; falls back to CPU.
            Sets _paddle_init_done=True when finished (pass or fail).
            Matches SerDet_PaddleOcr_02.py:
              PaddleOCR(use_textline_orientation=True, lang='en',
                        enable_mkldnn=False, ocr_version="PP-OCRv5", device="gpu")
            """
            import traceback
            print('[CAM2] _init_paddle thread started', flush=True)

            if not _PADDLE_OK:
                print('[CAM2] ❌ PaddleOCR not installed — '
                      'run: pip install paddleocr paddlepaddle-gpu', flush=True)
                self._paddle_init_failed = True
                self._paddle_init_done   = True
                return

            # Detect GPU
            use_gpu  = False
            gpu_name = 'CPU'
            try:
                import torch
                if torch.cuda.is_available():
                    torch.zeros(1, device='cuda:0')
                    use_gpu  = True
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f'[CAM2] PaddleOCR will use GPU: {gpu_name}', flush=True)
                else:
                    print('[CAM2] No CUDA — PaddleOCR will use CPU', flush=True)
            except Exception as _ge:
                print(f'[CAM2] ⚠️  GPU check: {_ge} — using CPU', flush=True)
                use_gpu = False

            # Load PaddleOCR (GPU first, then CPU retry)
            for _attempt, _dev in enumerate(['gpu' if use_gpu else 'cpu', 'cpu']):
                try:
                    print(f'[CAM2] PaddleOCR loading '
                          f'(attempt {_attempt+1}/2, device={_dev}) …', flush=True)
                    self.paddle_reader = _PaddleOCR(
                        use_textline_orientation=True,
                        lang='en',
                        enable_mkldnn=False,
                        ocr_version='PP-OCRv5',
                        device=_dev,
                        show_log=False,
                    )
                    print(f'[CAM2] ✅ PaddleOCR ready  device={_dev}  '
                          f'gpu_name={gpu_name if use_gpu else "CPU"}', flush=True)
                    self._paddle_init_done = True
                    return
                except Exception as _e:
                    print(f'[CAM2] ❌ PaddleOCR attempt {_attempt+1} failed: {_e}',
                          flush=True)
                    traceback.print_exc()
                    self.paddle_reader = None
                    if _attempt == 0 and use_gpu:
                        print('[CAM2] Retrying with CPU …', flush=True)
                    else:
                        break

            print('[CAM2] ❌ PaddleOCR failed on all attempts', flush=True)
            self._paddle_init_failed = True
            self._paddle_init_done   = True

        threading.Thread(target=_init_yolo, daemon=True).start()
        threading.Thread(target=_init_paddle, daemon=True).start()

        # ââ YOLO thread decoupled from main loop (fixes stream lag) ââ
        self._yolo_active        = False
        # _yolo_latest_frame: main loop writes, YOLO thread reads.
        # NOT cleared after reading â YOLO thread tracks its own last-
        # processed frame via _yolo_last_processed_id to detect new frames.
        self._yolo_latest_frame  = None
        self._yolo_last_id       = 0    # incremented by main loop each new frame
        self._yolo_processed_id  = 0    # last frame id processed by YOLO thread
        self._yolo_frame_lock    = threading.Lock()
        self._yolo_result_lock   = threading.Lock()
        self._yolo_latest_result = (None, 0, 0, 0, 0, False)

        # ââ Per-frame OCR queue (like video_test_03_cv2.py) âââââââââââ
        # Every YOLO detection â crop pushed here â OCR worker processes
        # immediately â votes â 2 matches = confirmed serial.
        # [FIX] maxsize=10: larger buffer so serial crops are not dropped when
        # PaddleOCR preprocessing takes >200 ms on Pi-class CPUs.
        # drop-oldest policy in _yolo_infer_thread prevents stale queue buildup.
        self._ocr_crop_queue    = _queue_mod.Queue(maxsize=0)   # unlimited â keep ALL SEQ1 crops
        self._scan_start_ts_ref = [0.0]   # for elapsed calc in YOLO thread

        # ââ serial.pt remains idle until the first SEQ1 activation.
        # We do not run YOLO on Camera-2 frames until SEQ1 begins.
        # This prevents Camera-2 from detecting serial regions before the
        # panel is actually placed and SEQ1 is active.
        self._yolo_active = False
        print("[CAM2] â¸ï¸ YOLO idle until start_ocr() is called for SEQ1")

        # ââ Start frame reader first, then YOLO, then OCR worker ââââââââââââ
        threading.Thread(target=self._main_loop,         daemon=True).start()
        # Give the reader a moment to connect and receive the first frame
        # before the YOLO thread tries to read _yolo_latest_frame.
        # On fast GPU systems this is negligible; on Pi it prevents a
        # 0.5 s window where YOLO thread sees None and does nothing useful.
        _t0 = time.time()
        while self._raw_frame is None and (time.time() - _t0) < 3.0:
            time.sleep(0.05)
        if self._raw_frame is not None:
            print("[CAM2] â First Camera-2 frame ready â starting YOLO + OCR threads")
        else:
            print("[CAM2] â ï¸  Camera-2 first frame not received in 3s â "
                  "starting threads anyway (will process once stream connects)")
        threading.Thread(target=self._yolo_infer_thread, daemon=True).start()
        threading.Thread(target=self._ocr_worker,        daemon=True).start()
        print("[CAM2] â All Camera-2 threads running â serial.pt active from first frame")

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  PUBLIC API
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def reset_for_new_panel(self):
        """
        Full reset for the next panel.
        Called by app_vision.reset_panel() before each new panel arrives.
        Clears all capture slots, OCR state, serial, folder, and scan flags
        so stale data from the previous panel never bleeds into the next one.
        """
        with self._lock:
            self.ocr_done           = False
            self.serial_number      = None
            self.ocr_buffer         = []
            self.day_votes          = Counter()
            self.code_votes         = Counter()
            self.letter_votes       = Counter()
            self.day_confirmed      = None
            self.code_confirmed     = None
            self.letter_confirmed   = None
            self.partial_serial     = None
            self.stable_count       = 0
            self._is_scanning       = False
            self.is_scanning        = False
            self.status             = "Waiting for new panel..."
            self.save_burst_frames  = True
            self._frames_saved      = 0
            self.intervals_saved    = set()
            self._capture_slots     = [None] * self.BEST_FRAME_COUNT
            self._slots_saved       = [False] * self.BEST_FRAME_COUNT
            self._ocr_buffer_frames = []
            self._best_frames_saved = False
            self._ocr_event.clear()
            self._ocr_running       = False
            self.panel_folder       = "."
            self.cam2_roi_path      = None
            self.cam2_raw_path      = None
            # Reset serial-appeared tracking
            self._appeared_frame1_saved = False
            self._appeared_frame1_path  = None
            self._appeared_best_sharp   = 0.0
            self._appeared_best_path    = None
            self._best_full_sharp       = 0.0

        # Clear latest detection so previous panel bbox disappears from stream
        with self._det_lock:
            self._latest_det    = None
            self._roi_fallback  = False

        # New panel â keep YOLO idle until the next SEQ1 activation.
        self._yolo_active = False
        while not self._ocr_crop_queue.empty():
            try: self._ocr_crop_queue.get_nowait()
            except: break
        self._scan_start_ts_ref[0] = time.time()
        print("[CAM2] reset_for_new_panel â â YOLO idle, queue cleared")

    def start_ocr(self):
        with self._lock:
            self.ocr_done           = False
            self.serial_number      = None
            self.ocr_buffer         = []
            self.day_votes          = Counter()
            self.code_votes         = Counter()
            self.letter_votes       = Counter()
            self.day_confirmed      = None
            self.code_confirmed     = None
            self.letter_confirmed   = None
            self.partial_serial     = None
            self.stable_count       = 0
            self._is_scanning       = True
            self.is_scanning        = True
            self.status             = "Scanning..."
            self.save_burst_frames  = True
            self._frames_saved      = 0
            self.intervals_saved    = set()
            self._capture_slots     = [None] * self.BEST_FRAME_COUNT
            self._slots_saved       = [False] * self.BEST_FRAME_COUNT
            self._ocr_buffer_frames = []
            self._best_frames_saved = False
            self._ocr_event.clear()      # clear any pending OCR signal
            self._ocr_running       = False
            # Reset serial-appeared tracking
            self._appeared_frame1_saved = False
            self._appeared_frame1_path  = None
            self._appeared_best_sharp   = 0.0
            self._appeared_best_path    = None
            self._best_full_sharp       = 0.0
            if (hasattr(self,'_last_good_ts_global')
                    and self._last_good_ts_global
                    and time.time()-self._last_good_ts_global > 5.0):
                self._force_reconnect = True

        # Activate YOLO detection for this panel
        self._yolo_active = True
        self._scan_start_ts_ref[0] = time.time()
        # Drain any stale crops from previous panel
        while not self._ocr_crop_queue.empty():
            try: self._ocr_crop_queue.get_nowait()
            except: break
        
        # Enhanced logging for diagnostics
        detector_status = "â" if self.serial_detector is not None else "â"
        ocr_status = "â" if self.paddle_reader is not None else "â"
        print(f"\n[CAM2] ð OCR INITIATED:")
        print(f"       YOLO Detector: {detector_status} {getattr(self.serial_detector, '_device', 'N/A') if self.serial_detector else 'Not loaded'}")
        print(f"       PaddleOCR:     {ocr_status} {'Ready' if self.paddle_reader else 'Not loaded'}")
        print(f"       Folder:        {self.panel_folder}")
        print(f"       Scanning:      ð¢ ACTIVE")
        print()

    def stop_scanning(self):
        with self._lock:
            self._is_scanning = False
            self.is_scanning  = False

    def stop_yolo_only(self):
        """
        Stop serial.pt inference (no new crops pushed to queue) but
        keep _is_scanning=True so the OCR worker drains existing queue.
        Called when SEQ1 completes: panel turned, serial no longer visible.
        """
        self._yolo_active = False
        print("[CAM2] serial.pt stopped (panel turned after SEQ1). "
              "OCR worker continues draining queued frames.")

    def stop_yolo_detection(self):
        """
        Called by app_vision.py after SEQ1 completes.
        Stops serial.pt from running on new Camera-2 frames.
        OCR worker continues running in background until serial confirmed.
        """
        self._yolo_active = False
        print("[CAM2] ð YOLO detection stopped â SEQ1 complete, "
              "OCR worker still running in background")

    def set_panel_folder(self, folder: str):
        """
        Assign the panel save folder.

        FIX-SEQ1: Immediately flushes any frames that were cached while the
        folder was not yet set (they would otherwise be silently dropped).
        """
        self.panel_folder = folder
        if not folder or folder == '.':
            return

        pending = getattr(self, '_pending_slot_frames', [])
        if not pending:
            return

        print(f'[CAM2] ð Folder set â flushing {len(pending)} pending slot(s)')
        for i, pd in enumerate(pending[: self.BEST_FRAME_COUNT]):
            if i >= self.BEST_FRAME_COUNT:
                break
            if self._slots_saved[i]:
                continue
            try:
                self._capture_slots[i] = {
                    'frame':   pd['frame'],
                    'crop':    pd['crop'],
                    'sharp':   pd['sharp'],
                    'x1': pd['x1'], 'y1': pd['y1'],
                    'x2': pd['x2'], 'y2': pd['y2'],
                    'elapsed': pd['elapsed'],
                }
                self._slots_saved[i] = True
                self._save_slot_to_disk(i, pd['frame'], pd['crop'], pd['sharp'])
                print(f'[CAM2] â Flushed pending slot {i+1}  sharp={pd["sharp"]:.0f}')
            except Exception as e:
                print(f'[CAM2] â ï¸  Error flushing slot {i+1}: {e}')
        self._pending_slot_frames = []
        print('[CAM2] â Pending frame flush complete')


    def disable_frame_saving(self):
        self.save_burst_frames = False

    def read(self):
        """Raw frame (no annotation) â kept for compat."""
        if self._cap is not None:
            return self._cap.read()
        return False, None

    def get_annotated_frame(self) -> np.ndarray | None:
        """Returns latest Camera-2 frame with bbox drawn (backward compat)."""
        frame, _ = self.get_annotated_frame_with_det()
        return frame

    def get_annotated_frame_with_det(self):
        """
        Returns (annotated_frame, det_or_None).

        LAG FIX: always read from _raw_frame (updated every 5 ms by main loop)
        instead of _latest_frame (only updated after YOLO inference, every 40+ ms).
        The bbox overlay (_latest_det) may be up to ~40 ms stale but the base
        frame is always the newest available â this is the correct trade-off.
        """
        # ââ Freshest base frame ââââââââââââââââââââââââââââââââââââââââ
        with self._raw_frame_lock:
            frame = (self._raw_frame.copy()
                     if self._raw_frame is not None else None)

        if frame is None:
            # _raw_frame not populated yet â try _latest_frame fallback
            with self._det_lock:
                frame = (self._latest_frame.copy()
                         if self._latest_frame is not None else None)
            if frame is None:
                return None, None

        # ââ Latest bbox from YOLO (may be ~40 ms stale â acceptable) ââââ
        with self._det_lock:
            det = self._latest_det

        if det is not None and self.serial_detector is not None:
            dbg = getattr(self, '_annotate_debug_count', 0)
            if dbg % 60 == 0:
                print(f"[CAM2-UI] bbox conf={det[4]:.2f}")
            self._annotate_debug_count = dbg + 1
            return self.serial_detector.annotate(frame, det, label="serial"), det

        return frame, None

    def get_latest_detection(self):
        """Returns latest (x1,y1,x2,y2,conf) or None."""
        with self._det_lock:
            return self._latest_det

    def get_status(self):        return self.status
    def get_serial_number(self): return self.serial_number
    def is_done(self):           return self.ocr_done

    def get_voting_state(self):
        """Return current vote counts for UI display (matches reference output)."""
        return {
            'day_votes':    dict(self.day_votes),
            'code_votes':   dict(self.code_votes),
            'letter_votes': dict(self.letter_votes),
            'day_confirmed':    self.day_confirmed,
            'code_confirmed':   self.code_confirmed,
            'letter_confirmed': self.letter_confirmed,
            'final_serial': self.serial_number,
        }

    def get_last_frame(self):
        with self._lock:
            return (self.last_frame.copy()
                    if self.last_frame is not None else None)

    def is_burst_complete(self):
        with self._lock:
            # Burst is complete once OCR confirmed (interval frames removed)
            return self.ocr_done

    def stop(self):
        self.running = False
        if hasattr(self._cap,'release'): self._cap.release()
        if self.serial_detector:         self.serial_detector.close()

    def capture_single_audit_frame(self, label):
        """
        Milestone capture (20%, 50%, 90%) â saves the annotated frame
        so the serial bbox is visible in audit photos.
        """
        if not self.panel_folder or self.panel_folder == ".":
            return False
        frame = self.get_annotated_frame()
        if frame is not None:
            path = os.path.join(self.panel_folder,
                                f"CAM2_Progress_{label}.jpg")
            ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
            if ok:
                print(f"[CAM2] â Progress {label}% â "
                      f"{os.path.basename(path)}")
            return ok
        return False

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  YOLO INFERENCE ON EVERY FRAME  (always-on, single call)
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _run_yolo_on_frame(self, frame: np.ndarray):
        """
        Runs serial.pt ONCE per frame.
        Updates _latest_det + _latest_frame for UI annotation.
        Returns (crop, x1, y1, x2, y2, yolo_detected).
        NO ROI fallback â only real YOLO detections are used.
        If no detection â updates frame for UI, returns yolo_detected=False.
        """
        if self.serial_detector is None:
            # No YOLO loaded â show clean frame, no bbox
            with self._det_lock:
                self._latest_frame = frame.copy()
                self._latest_det   = None
                self._roi_fallback = False
            return None, 0, 0, 0, 0, False

        # Run inference â no lock held during inference
        try:
            dets = self.serial_detector.detect(frame)
        except Exception as e:
            print(f"[CAM2-YOLO] detect() error: {e}")
            dets = []

        # Debug logging
        dbg = getattr(self, '_serial_debug_count', 0)
        if dets:
            if dbg % 10 == 0:
                print(f"[CAM2-YOLO] â Serial detected: "
                      f"{len(dets)} boxes, conf={dets[0][4]:.2f}")
        else:
            if dbg % 60 == 0:
                print("[CAM2-YOLO] No serial detection")
        self._serial_debug_count = dbg + 1

        # ââ No detection â clean frame, no bbox in UI ââââââââââ
        if not dets:
            with self._det_lock:
                self._latest_frame = frame.copy()
                self._latest_det   = None
                self._roi_fallback = False
            return None, 0, 0, 0, 0, False

        # ââ YOLO detection â crop exactly the bbox region âââââââ
        ih, iw         = frame.shape[:2]
        x1, y1, x2, y2, conf = dets[0]
        p   = self.serial_detector.CROP_PAD_PX
        cx1 = max(0,  x1 - p)
        cy1 = max(0,  y1 - p)
        cx2 = min(iw, x2 + p)
        cy2 = min(ih, y2 + p)
        crop = frame[cy1:cy2, cx1:cx2].copy()

        if crop.size == 0:
            with self._det_lock:
                self._latest_frame = frame.copy()
                self._latest_det   = None
                self._roi_fallback = False
            return None, 0, 0, 0, 0, False

        # Atomic UI update â green bbox only, no ROI fallback
        with self._det_lock:
            self._latest_frame = frame.copy()
            self._latest_det   = dets[0]
            self._roi_fallback = False

        return crop, cx1, cy1, cx2, cy2, True

    def _get_roi_crop(self, frame: np.ndarray) -> np.ndarray | None:
        """
        FIX-11: Generate a fallback ROI crop when YOLO detection fails.
        Uses a fixed region in the frame (right 1/3, middle third height).
        Returns the cropped region or None if frame too small.
        """
        h, w = frame.shape[:2]
        if h < 100 or w < 150:
            return None
        
        # Right 1/3 of frame, middle 1/3 vertically (typically where serial is)
        x1_roi = int(w * 0.60)  # Start at 60% from left
        x2_roi = int(w * 0.98)  # End near right edge
        y1_roi = int(h * 0.35)  # Start at 35% from top
        y2_roi = int(h * 0.65)  # End at 65% from top
        
        roi_crop = frame[y1_roi:y2_roi, x1_roi:x2_roi]
        if roi_crop.size == 0:
            return None
        
        self._roi_fallback = True
        return roi_crop

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  PROGRESSIVE CAPTURE SLOTS
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _try_fill_slot(self, frame:np.ndarray, crop:np.ndarray,
                        x1:int, y1:int, x2:int, y2:int,
                        elapsed:float):
        """
        Try to fill the next unfilled capture slot.
        A slot is fillable if elapsed >= its target time AND
        the crop passes sharpness threshold.
        Only called when yolo_detected=True.

        FIX-SEQ1: When panel_folder is not yet set we cache up to 3 frames
        instead of silently dropping them.  set_panel_folder() will flush
        the cache immediately when the folder is assigned.
        """
        folder = self.panel_folder
        if not folder or folder == ".":
            # ââ Cache frame for later â folder not set yet ââââââââ
            if not hasattr(self, '_pending_slot_frames'):
                self._pending_slot_frames = []
            if len(self._pending_slot_frames) < self.BEST_FRAME_COUNT:
                sharp_now = sharpness_score(crop)
                self._pending_slot_frames.append({
                    'frame': frame.copy(), 'crop': crop.copy() if crop is not None else None,
                    'sharp': sharp_now,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'elapsed': elapsed,
                })
                print(f'[CAM2] â±ï¸  Pending slot cached (no folder yet) '
                      f'â queue={len(self._pending_slot_frames)}  sharp={sharp_now:.0f}')
            return

        sharp     = sharpness_score(crop)
        threshold = (self.MIN_SHARPNESS_ROI
                     if self._roi_fallback else self.MIN_SHARPNESS)
        if sharp < threshold:
            return   # blurry â wait for a better frame

        for idx, target in enumerate(self.CAPTURE_TARGETS):
            if self._slots_saved[idx]:
                continue   # already filled
            if elapsed < target:
                continue   # not time yet

            # Fill this slot
            self._capture_slots[idx] = {
                'frame': frame.copy(),
                'crop':  crop,
                'sharp': sharp,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'elapsed': elapsed,
            }
            self._slots_saved[idx] = True
            self._save_slot_to_disk(idx, frame, crop, sharp)
            break   # fill one slot per call

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  SERIAL-APPEARED FRAMES  (main folder, 2 frames only)
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _frame_is_valid(self, frame: np.ndarray, min_sharp: float = 5.0) -> bool:
        """Return True if frame is non-empty, non-black, and sharp enough."""
        if frame is None or frame.size == 0:
            return False
        # Brightness check â reject black/near-black frames
        mean_val = float(frame.mean())
        if mean_val < 8.0:
            return False
        # Sharpness check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return sharp >= min_sharp

    def _save_serial_appeared_frame(self, frame: np.ndarray, crop: np.ndarray,
                                     x1: int, y1: int, x2: int, y2: int):
        """
        Save exactly 2 frames to the MAIN panel folder when serial_number class
        first appears:
          â¢ Appeared_1  â first detection frame
          â¢ Appeared_Best â sharpest detection frame seen (updated until OCR done)
        No blurry / empty frames accepted.
        """
        folder = self.panel_folder
        if not folder or folder == ".":
            return

        if self.ocr_done:          # serial already confirmed â no more saves
            return

        if not self._frame_is_valid(frame, min_sharp=5.0):
            return

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        ts    = datetime.now().strftime("%H%M%S")
        os.makedirs(folder, exist_ok=True)

        # Frame 1 â first moment serial class appears
        if not getattr(self, '_appeared_frame1_saved', False):
            path1 = os.path.join(folder, f"CAM2_Serial_Appeared_1_{ts}.jpg")
            if cv2.imwrite(path1, frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
                self._appeared_frame1_saved = True
                self._appeared_frame1_path  = path1
                print(f"[CAM2] ð¸ Serial appeared (frame 1) â main folder  "
                      f"sharp={sharp:.0f}")

        # Frame Best â keep updating with sharpest frame until OCR confirmed
        best_sharp = getattr(self, '_appeared_best_sharp', 0.0)
        if sharp > best_sharp:
            path_best = os.path.join(folder, f"CAM2_Serial_Appeared_Best_{ts}.jpg")
            # Remove previous best file to avoid accumulation
            old_best = getattr(self, '_appeared_best_path', None)
            if old_best and os.path.exists(old_best) and old_best != path_best:
                try: os.remove(old_best)
                except Exception: pass
            if cv2.imwrite(path_best, frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
                self._appeared_best_sharp = sharp
                self._appeared_best_path  = path_best
                print(f"[CAM2] ð¸ Serial appeared (best updated) sharp={sharp:.0f}")

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  SERIAL_CAPTURES SUBFOLDER  (structured 4-file save)
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _save_slot_to_disk(self, idx: int, frame: np.ndarray,
                            crop: np.ndarray, sharp: float):
        """
        Save serial detection frames to serial_captures/ subfolder.

        Slot 1 (first detection):
          â¢ CAM2_ROI_Annotated_HHMMSS.jpg  â full frame with bbox drawn
          â¢ CAM2_Full_Frame_HHMMSS.jpg     â clean full frame
          â¢ CAM2_Crop_HHMMSS.jpg           â exact YOLO detection crop
          â¢ CAM2_Best_Full_HHMMSS.jpg      â copy of clean full (best so far)

        Slot 2 (second detection):
          â¢ Overwrites CAM2_Best_Full if this slot is sharper
          â¢ Saves its own CAM2_Crop for voting diversity

        No more than 2 slots saved (BEST_FRAME_COUNT = 2).
        Quality-gated: blurry / empty frames rejected.
        """
        folder = self.panel_folder
        if not folder or folder == ".":
            return

        # Quality gate
        if not self._frame_is_valid(frame, min_sharp=5.0):
            print(f"[CAM2] â ï¸  Slot {idx+1} rejected â invalid frame")
            return
        if crop is None or crop.size == 0:
            print(f"[CAM2] â ï¸  Slot {idx+1} rejected â empty crop")
            return

        n   = idx + 1
        ts  = datetime.now().strftime("%H%M%S")
        serial_dir = os.path.join(folder, "serial_captures")
        os.makedirs(serial_dir, exist_ok=True)

        if n == 1:
            # ââ ROI Annotated (full frame with detection box) âââââââââ
            annotated = frame.copy()
            cv2.rectangle(annotated, (x1 := self._capture_slots[idx]['x1'],
                                       y1 := self._capture_slots[idx]['y1']),
                          (self._capture_slots[idx]['x2'],
                           self._capture_slots[idx]['y2']),
                          (0, 255, 0), 2)
            cv2.putText(annotated, "serial_number",
                        (self._capture_slots[idx]['x1'],
                         max(self._capture_slots[idx]['y1'] - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            roi_path = os.path.join(serial_dir, f"CAM2_ROI_Annotated_{ts}.jpg")
            cv2.imwrite(roi_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 98])

            # ââ Clean full frame ââââââââââââââââââââââââââââââââââââââ
            full_path = os.path.join(serial_dir, f"CAM2_Full_Frame_{ts}.jpg")
            cv2.imwrite(full_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 98])

            # ââ Exact YOLO crop (Unpadded) âââââââââââââââââââââââââââââ
            x1_exact, y1_exact = self._capture_slots[idx]['x1'], self._capture_slots[idx]['y1']
            x2_exact, y2_exact = self._capture_slots[idx]['x2'], self._capture_slots[idx]['y2']
            unpadded_crop = frame[y1_exact:y2_exact, x1_exact:x2_exact].copy()
            exact_path = os.path.join(serial_dir, f"CAM2_Exact_Crop_{ts}.jpg")
            if unpadded_crop.size > 0:
                cv2.imwrite(exact_path, unpadded_crop, [cv2.IMWRITE_JPEG_QUALITY, 98])

            # ââ Padded YOLO crop ââââââââââââââââââââââââââââââââââââââââ
            crop_path = os.path.join(serial_dir, f"CAM2_Padded_Crop_{ts}.jpg")
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 98])

            # ââ Best full frame (slot 1 is initial best) ââââââââââââââ
            best_path = os.path.join(serial_dir, "CAM2_Best_Full.jpg")
            cv2.imwrite(best_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
            self._best_full_sharp = sharp

            # Track for PDF linking
            self.cam2_raw_path = full_path
            self.cam2_roi_path  = crop_path

            print(f"[CAM2] ð¸ serial_captures/ slot 1 â sharp={sharp:.0f}")
            print(f"       ROI annotated : {os.path.basename(roi_path)}")
            print(f"       Full frame    : {os.path.basename(full_path)}")
            print(f"       Crop          : {os.path.basename(crop_path)}")
            print(f"       Best full     : CAM2_Best_Full.jpg")

        else:
            # ââ Slot 2: extra crops for voting diversity ââââââââââââââââ
            x1_exact, y1_exact = self._capture_slots[idx]['x1'], self._capture_slots[idx]['y1']
            x2_exact, y2_exact = self._capture_slots[idx]['x2'], self._capture_slots[idx]['y2']
            unpadded_crop2 = frame[y1_exact:y2_exact, x1_exact:x2_exact].copy()
            exact2_path = os.path.join(serial_dir, f"CAM2_Exact_Crop_2_{ts}.jpg")
            if unpadded_crop2.size > 0:
                cv2.imwrite(exact2_path, unpadded_crop2, [cv2.IMWRITE_JPEG_QUALITY, 98])

            crop2_path = os.path.join(serial_dir, f"CAM2_Padded_Crop_2_{ts}.jpg")
            cv2.imwrite(crop2_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 98])

            # ââ Update Best_Full if this slot is sharper ââââââââââââââ
            best_path = os.path.join(serial_dir, "CAM2_Best_Full.jpg")
            prev_sharp = getattr(self, '_best_full_sharp', 0.0)
            if sharp > prev_sharp:
                cv2.imwrite(best_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
                self._best_full_sharp = sharp
                print(f"[CAM2] ð¸ serial_captures/ slot 2 â Best_Full updated "
                      f"sharp={sharp:.0f}")
            else:
                print(f"[CAM2] ð¸ serial_captures/ slot 2 â extra crop saved "
                      f"sharp={sharp:.0f}")

        # Run live OCR on each slot immediately
        if _PADDLE_OK and self.paddle_reader is not None:
            print(f"[CAM2] ð Live OCR on slot {n} ...")
            try:
                # Use exact reference _ocr_pipeline (2DFilter then Otsu)
                slot_serial = self._ocr_pipeline(crop)
                if slot_serial:
                    print(f"[CAM2] ð¬ Slot {n} reading: '{slot_serial}'")
                    with self._lock:
                        self.partial_serial = slot_serial
                        self.status = f"Slot {n}: {slot_serial}"
                else:
                    print(f"[CAM2] â ï¸  Slot {n}: no serial extracted yet")

                # Save slot OCR detail to serial_captures/
                ocr_txt = os.path.join(serial_dir, f"Slot_{n}_OCR.txt")
                with open(ocr_txt, "w", encoding="utf-8") as f:
                    f.write(f"Slot {n} OCR  sharp={sharp:.0f}  ts={ts}\n")
                    f.write("\n".join(live_lines))
                    f.write(f"\n\nExtracted: {slot_serial}\n")
            except Exception as e:
                print(f"[CAM2] Live OCR error: {e}")

    def _all_slots_filled(self) -> bool:
        return all(self._slots_saved)

    def _get_best_frames_for_ocr(self) -> list:
        """Return filled slots sorted by sharpness (best first)."""
        filled = [s for s in self._capture_slots if s is not None]
        return sorted(filled, key=lambda e: e['sharp'], reverse=True)

    # ─────────────────────────────────────────────────────────
    #  OCR   (voting-based)
    # ─────────────────────────────────────────────────────────

    def _run_ocr_with_voting(self, crop):
        """Run preprocessing variants; vote per serial position.
        Returns final serial string or None.
        """
        from collections import Counter
        CONFIRM_THRESHOLD = 3

        # Preprocessing variants
        variants_to_try = list(preprocess_engraved_metal(crop))
        day_confirmed    = None
        code_confirmed   = None
        letter_confirmed = None
        day_votes    = Counter()
        code_votes   = Counter()
        letter_votes = Counter()

        for variant_idx, (name, pp_img) in enumerate(variants_to_try, start=1):
            raw = self._run_paddle_ocr(pp_img)
            if not raw:
                print(f"[CAM2-OCR-VOTE]   [{name:10}] (no text)")
                continue
            
            # Validate and correct
            validated = _correct_serial(raw)
            if not validated:
                print(f"[CAM2-OCR-VOTE]   [{name:10}] {repr(raw):12} â (validation failed)")
                continue
            
            # Extract positions
            day, code, letter = self._extract_day_code_letter(validated)
            if day is None or code is None or letter is None:
                print(f"[CAM2-OCR-VOTE]   [{name:10}] {repr(raw):12} â (extraction failed)")
                continue
            
            # Count votes for each position
            day_votes[day] += 1
            code_votes[code] += 1
            letter_votes[letter] += 1
            
            # Show vote status
            day_count = day_votes[day] if day_confirmed is None else 0
            code_count = code_votes[code] if code_confirmed is None else 0
            letter_count = letter_votes[letter] if letter_confirmed is None else 0
            
            status_str = f"[CAM2-OCR-VOTE]   [{name:10}] {repr(raw):12} â {day}/{code}/{letter}"
            votes_str = f"  Day={day_count}/{CONFIRM_THRESHOLD}"
            if code_confirmed is None:
                votes_str += f" Code={code_count}/{CONFIRM_THRESHOLD}"
            if letter_confirmed is None:
                votes_str += f" Letter={letter_count}/{CONFIRM_THRESHOLD}"
            print(status_str + f"  ({votes_str})")
            
            # Freeze position when it reaches threshold
            if day_confirmed is None and day_votes[day] >= CONFIRM_THRESHOLD:
                day_confirmed = day
                print(f"[CAM2-OCR-VOTE]   â­ DAY FROZEN: '{day}' (got {CONFIRM_THRESHOLD} votes)")
            
            if code_confirmed is None and code_votes[code] >= CONFIRM_THRESHOLD:
                code_confirmed = code
                print(f"[CAM2-OCR-VOTE]   â­ CODE FROZEN: '{code}' (got {CONFIRM_THRESHOLD} votes)")
            
            if letter_confirmed is None and letter_votes[letter] >= CONFIRM_THRESHOLD:
                letter_confirmed = letter
                print(f"[CAM2-OCR-VOTE]   â­ LETTER FROZEN: '{letter}' (got {CONFIRM_THRESHOLD} votes)")
            
            # All positions confirmed? EARLY EXIT!
            if day_confirmed and code_confirmed and letter_confirmed:
                print(f"[CAM2-OCR-VOTE] â EARLY EXIT at variant {variant_idx}/{len(variants_to_try)}")
                break
        
        # Build final result if all confirmed
        if day_confirmed and code_confirmed and letter_confirmed:
            from datetime import datetime as _dt
            now = _dt.now()
            mm = f'{now.month:02d}'
            yy = f'{now.year % 100:02d}'
            final = day_confirmed + mm + yy + code_confirmed + letter_confirmed
            print(f"[CAM2-OCR-VOTE] ð¯ FINAL SERIAL: {final} (fully voted)")
            print(f"  Day:    {day_confirmed}  ({day_votes[day_confirmed]} votes)")
            print(f"  Code:   {code_confirmed}  ({code_votes[code_confirmed]} votes)")
            print(f"  Letter: {letter_confirmed}  ({letter_votes[letter_confirmed]} votes)\n")
            return final
        
        # FALLBACK: Use best guess for unconfirmed positions
        print(f"\n[CAM2-OCR-VOTE] â¹ï¸  Fallback mode - using best guesses:")
        if not day_confirmed:
            best_day = max(day_votes, key=day_votes.get, default=None)
            if best_day:
                day_confirmed = best_day
                print(f"  Day:    {best_day} (best of {day_votes[best_day]} votes)")
        else:
            print(f"  â Day:    {day_confirmed} ({day_votes[day_confirmed]} votes)")
        
        if not code_confirmed:
            best_code = max(code_votes, key=code_votes.get, default=None)
            if best_code:
                code_confirmed = best_code
                print(f"  Code:   {best_code} (best of {code_votes[best_code]} votes)")
        else:
            print(f"  â Code:   {code_confirmed} ({code_votes[code_confirmed]} votes)")
        
        if not letter_confirmed:
            best_letter = max(letter_votes, key=letter_votes.get, default=None)
            if best_letter:
                letter_confirmed = best_letter
                print(f"  Letter: {best_letter} (best of {letter_votes[best_letter]} votes)")
        else:
            print(f"  â Letter: {letter_confirmed} ({letter_votes[letter_confirmed]} votes)")
        
        # If we got best guesses for all positions, use them
        if day_confirmed and code_confirmed and letter_confirmed:
            from datetime import datetime as _dt
            now = _dt.now()
            mm = f'{now.month:02d}'
            yy = f'{now.year % 100:02d}'
            final = day_confirmed + mm + yy + code_confirmed + letter_confirmed
            print(f"[CAM2-OCR-VOTE] ð USING BEST GUESS: {final}\n")
            return final
        
        print(f"[CAM2-OCR-VOTE] â NO RESULT\n")
        return None

    def _ocr_pipeline(self, crop) -> str:
        """Run PaddleOCR on a YOLO-padded serial crop.

        Pipeline (matches SerDet_PaddleOcr_02.py):
          1. crop  = serial.pt bbox + CROP_PAD_PX padding (already done)
          2. Resize 2× (cv2.INTER_CUBIC) for better OCR accuracy
          3. PaddleOCR.predict() on the resized BGR image
          4. Extract rec_texts / rec_scores, filter score >= 0.3
          5. Validate with _correct_serial() → 10-char DDMMYY+3digits+L
          6. Save raw crop + PaddleOCR result to serial_captures/
             - last_raw_crop.jpg      (the padded YOLO crop before resize)
             - last_ocr_input.jpg     (the 2× resized image sent to OCR)
             - paddleocr_predictions.txt  (appended every frame)

        Returns validated serial string or None.
        """
        from datetime import datetime as _dtp

        # ── resolve save dir ─────────────────────────────────────────
        if self.panel_folder and self.panel_folder not in ('.', ''):
            sd = os.path.join(self.panel_folder, "serial_captures")
        else:
            sd = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "serial_captures")
        try:
            os.makedirs(sd, exist_ok=True)
        except Exception:
            sd = None

        # ── save raw padded crop ──────────────────────────────────────
        if sd:
            try:
                cv2.imwrite(os.path.join(sd, "last_raw_crop.jpg"),
                            crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            except Exception:
                pass

        # ── resize 2× (SerDet_PaddleOcr_02.py line: cv2.resize 2× cubic) ──
        ch, cw = crop.shape[:2]
        ocr_input = cv2.resize(crop, (2 * cw, 2 * ch),
                               interpolation=cv2.INTER_CUBIC)

        if sd:
            try:
                cv2.imwrite(os.path.join(sd, "last_ocr_input.jpg"),
                            ocr_input, [cv2.IMWRITE_JPEG_QUALITY, 95])
            except Exception:
                pass

        # ── PaddleOCR predict ─────────────────────────────────────────
        ts = _dtp.now().strftime("%H:%M:%S.%f")[:-3]
        log = [f"\n{'─'*64}",
               f"[{ts}]  raw={cw}×{ch}  ocr_input={2*cw}×{2*ch}  "
               f"reader={'READY' if self.paddle_reader else 'NONE'}"]

        raw_text, max_score = self._run_paddle_ocr(ocr_input)

        hit = _correct_serial(raw_text) if raw_text else None

        if hit:
            log.append(f"  ✅  raw={repr(raw_text):<26}  score={max_score:.3f}  → {hit}")
            print(f"[CAM2-OCR] ✅ PaddleOCR: {repr(raw_text)} score={max_score:.3f} → {hit}",
                  flush=True)
        elif raw_text:
            log.append(f"  ❌  raw={repr(raw_text):<26}  score={max_score:.3f}  (no match)")
            print(f"[CAM2-OCR] ❌ PaddleOCR: {repr(raw_text)} score={max_score:.3f} no match",
                  flush=True)
        else:
            log.append(f"  ──  (no text returned)")
            print(f"[CAM2-OCR] ── PaddleOCR: (no text)", flush=True)

        log.append(f"  RESULT → {hit or 'None'}")

        # ── append to predictions log ─────────────────────────────────
        if sd:
            try:
                with open(os.path.join(sd, "paddleocr_predictions.txt"),
                          "a", encoding="utf-8") as fh:
                    fh.write("\n".join(log) + "\n")
            except Exception as _e:
                print(f"[CAM2-OCR] predictions log write failed: {_e}", flush=True)

        return hit


    def _run_paddle_ocr(self, img) -> tuple:
        """Run PaddleOCR on one image (BGR or grayscale).

        Matches SerDet_PaddleOcr_02.py:
          result = reader.predict(in_img)
          result = result[0]
          result['rec_texts']   → list of text strings
          result['rec_scores']  → list of confidence floats

        Returns (joined_text, max_confidence).
        Logs every call to serial_captures/paddleocr_call_log.txt.
        """
        from datetime import datetime as _dtp
        ts  = _dtp.now().strftime("%H:%M:%S.%f")[:-3]
        ih, iw = img.shape[:2] if hasattr(img, 'shape') else (0, 0)

        # Resolve log path (fallback to script dir)
        _ld = (os.path.join(self.panel_folder, "serial_captures")
               if self.panel_folder and self.panel_folder not in ('.', '')
               else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "serial_captures"))
        try:
            os.makedirs(_ld, exist_ok=True)
            _lp = os.path.join(_ld, "paddleocr_call_log.txt")
        except Exception:
            _lp = None

        def _wlog(lines):
            if not _lp:
                return
            try:
                with open(_lp, "a", encoding="utf-8") as _f:
                    _f.write("\n".join(lines) + "\n")
            except Exception:
                pass

        if not _PADDLE_OK or self.paddle_reader is None:
            _wlog([f"[{ts}] img={iw}×{ih}  SKIPPED "
                   f"(PADDLE_OK={_PADDLE_OK} "
                   f"reader={'None' if self.paddle_reader is None else 'OK'})"])
            return ('', 0.0)

        # Ensure BGR (PaddleOCR expects BGR or RGB, not grayscale)
        if img.ndim == 2:
            img_in = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_in = img

        try:
            result = self.paddle_reader.predict(img_in)
        except Exception as e:
            _wlog([f"[{ts}] img={iw}×{ih}  EXCEPTION: {e}"])
            print(f"[CAM2-OCR] PaddleOCR exception: {e}", flush=True)
            return ('', 0.0)

        # ── Parse result (SerDet_PaddleOcr_02.py pattern) ───────────
        # result is a list; result[0] is a dict with rec_texts/rec_scores
        log = [f"[{ts}] img={iw}×{ih}  result_len={len(result) if result else 0}"]

        if not result:
            log.append("  (empty result)")
            _wlog(log)
            return ('', 0.0)

        item = result[0] if isinstance(result, list) else result
        texts  = []
        scores = []

        if isinstance(item, dict):
            rec_t = item.get('rec_texts', [])
            rec_s = item.get('rec_scores', [])
            if isinstance(rec_t, str):
                rec_t = [rec_t]
                rec_s = [rec_s] if not isinstance(rec_s, list) else rec_s
        elif isinstance(item, list):
            # Older API: [[bbox, text, conf], ...]
            rec_t = [x[1] for x in item if len(x) >= 3]
            rec_s = [x[2] for x in item if len(x) >= 3]
        else:
            rec_t, rec_s = [], []

        for i, (t, s) in enumerate(zip(rec_t, rec_s)):
            flag = "✓" if s >= 0.3 else "✗"
            log.append(f"  [{i}] text={repr(str(t)):<22} score={s:.3f} {flag}")
            if s >= 0.3:
                texts.append(str(t).strip())
                scores.append(float(s))

        joined    = ''.join(texts)
        max_score = max(scores) if scores else 0.0
        log.append(f"  → joined={repr(joined)}  max_score={max_score:.3f}  "
                   f"({len(texts)}/{len(rec_t)} accepted)")
        _wlog(log)
        return (joined, max_score)


    def _extract_serial(self, texts):
        """Legacy: used by slot live-OCR. Kept for compat."""
        for t in texts:
            hit = _correct_serial(str(t))
            if hit:
                return hit
        return None
    def _preprocess_crop(self, crop) -> list:
        """
        4x Upscale -> 7 Variants (2DFilter, Otsu, DoG, MG, Original, CLAHE, Adaptive)
        """
        import cv2
        import os
        import numpy as np

        h, w = crop.shape[:2]
        factor = getattr(self, 'OCR_UPSCALE_FACTOR', 4)
        up = cv2.resize(crop, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
        
        sd = os.path.join(self.panel_folder, "serial_captures") if self.panel_folder and self.panel_folder != "." else None
        if sd:
            os.makedirs(sd, exist_ok=True)
            try:
                cv2.imwrite(os.path.join(sd, "CAM2_Original_Crop.jpg"), crop)
                cv2.imwrite(os.path.join(sd, "CAM2_Upscaled.jpg"), up)
            except: pass

        grey = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY) if up.ndim == 3 else up.copy()
        variants = []

        k1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.int16)
        filt = cv2.filter2D(grey, -1, k1)
        variants.append(("2DFilter", filt))

        k2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.int16)
        sharp = cv2.filter2D(grey, -1, k2)
        _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(("Otsu", otsu))

        if sd:
            try:
                for name, img in variants:
                    cv2.imwrite(os.path.join(sd, f"CAM2_PP_{name}.jpg"), img)
            except: pass

        return variants

    def _run_ocr_on_slots(self) -> str | None:
        """
        OCR on capture slots with POSITION-BY-POSITION VOTING.
        
        â Each crop: run PaddleOCR on YOLO padded crop (2x resize)
        â Validate each result (day 01-31, code 3 digits, letter A-D)
        â Vote on each position independently  
        â Freeze when any position gets 3 matching votes
        â Return final serial only when ALL positions confirmed
        
        Much more robust than "first-match-wins" on factory floors
        with variable lighting/angles.
        """
        best = self._get_best_frames_for_ocr()
        if not best:
            return None

        all_serials = []
        ocr_details = []

        for idx, entry in enumerate(best, start=1):
            crop = entry['crop']
            h, w = crop.shape[:2]
            print(f"\n[CAM2-OCR] ââ Frame {idx} (sharpness={entry['sharp']:.0f}, "
                  f"size={w}Ã{h}) ââ")
            
            ocr_details.append(f"\n=== FRAME {idx} ===")
            ocr_details.append(f"Sharpness: {entry['sharp']:.0f}")
            ocr_details.append(f"Crop size: {w}Ã{h}")

            # Run voting-based OCR on this slot's crop
            serial = self._run_ocr_with_voting(crop)
            
            print(f"[CAM2-OCR] ââ Frame {idx} result: {serial or '(no vote consensus)'}  ââ\n")
            ocr_details.append(f"Frame result: '{serial}'")
            
            if serial:
                all_serials.append(serial)

        # If we got confirmed results from any frames, use the first one
        # (they should all be the same if voting worked correctly)
        final_serial = all_serials[0] if all_serials else None
        
        # Save audit trail
        if final_serial and self.panel_folder and self.panel_folder != ".":
            try:
                details_file = os.path.join(self.panel_folder, "paddleocr_readings.txt")
                with open(details_file, "w", encoding="utf-8") as f:
                    f.write(f"PaddleOCR Voting-Based Readings\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Method: Position-by-position voting (3 votes per position)\n")
                    f.write(f"Final Serial: {final_serial}\n")
                    f.write("\n".join(ocr_details))
                    f.write(f"\n\n=== FINAL RESULT ===\n{final_serial}\n")
                print(f"[CAM2-OCR] â Voting readings saved â paddleocr_readings.txt")
            except Exception as e:
                print(f"[CAM2-OCR] Error saving readings: {e}")
        
        return final_serial

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  CONFIRM + SAVE RESULT
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _rename_folder_with_serial(self, folder: str, serial: str) -> str:
        """
        Rename panel folder and any PDF inside it to include serial number.
        e.g. SSP-SEQ_191459 â 123456789A_191459
        Returns new folder path (or original if rename fails).
        """
        if not folder or folder == "." or not os.path.isdir(folder):
            return folder
        try:
            parent   = os.path.dirname(folder)
            basename = os.path.basename(folder)
            # Replace leading token (SSP-SEQ or anything before _) with serial
            parts    = basename.split("_", 1)
            new_name = f"{serial}_{parts[1]}" if len(parts) == 2 else serial
            new_folder = os.path.join(parent, new_name)

            # Rename any PDF inside before renaming folder
            for f in os.listdir(folder):
                if f.endswith(".pdf"):
                    old_pdf = os.path.join(folder, f)
                    pdf_parts = f.rsplit("_", 1)
                    new_pdf_name = (f"{serial}_{pdf_parts[1]}"
                                   if len(pdf_parts) == 2
                                   else f"{serial}.pdf")
                    try:
                        os.rename(old_pdf,
                                  os.path.join(folder, new_pdf_name))
                        print(f"[CAM2] ð PDF renamed â {new_pdf_name}")
                    except Exception as pe:
                        print(f"[CAM2] PDF rename failed: {pe}")

            os.rename(folder, new_folder)
            print(f"[CAM2] ð Folder renamed â {new_name}")
            return new_folder
        except Exception as e:
            print(f"[CAM2] Folder rename failed: {e}")
            return folder

    def _confirm_serial(self, serial: str):
        with self._lock:
            if self.ocr_done: return
            self.ocr_done      = True
            self.serial_number = serial
            self.status        = f"Done: {serial}"
            self._is_scanning  = False
            self.is_scanning   = False

        print(f"[CAM2] â Serial CONFIRMED: {serial}")

        folder = self.panel_folder
        if folder and folder != ".":
            try:
                os.makedirs(folder, exist_ok=True)
                with open(os.path.join(folder, "serial_ocr_result.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(f"Serial:    {serial}\n"
                            f"Timestamp: {datetime.now().isoformat()}\n"
                            f"Method:    serial.pt + PaddleOCR "
                            f"(2DFilter+Otsu, 3-slot)\n")
            except Exception:
                pass

            # ââ Rename folder + PDF with confirmed serial ââââââââââ
            new_folder = self._rename_folder_with_serial(folder, serial)
            with self._lock:
                self.panel_folder = new_folder

        if self.on_serial_detected:
            try: self.on_serial_detected(serial)
            except Exception as e:
                print(f"[CAM2] Callback error: {e}")

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  INTERVAL FRAMES  (PDF audit)
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _save_interval_frame(self, frame, num, saved_set):
        for attempt in range(1,4):
            try:
                if not self.panel_folder or self.panel_folder == ".": return
                os.makedirs(self.panel_folder, exist_ok=True)
                ts = datetime.now().strftime("%H%M%S")
                fp = os.path.join(self.panel_folder,
                                  f"SSP-SEQ_OCR_Frame_{num}_{ts}.jpg")
                if cv2.imwrite(fp, frame):
                    saved_set.add(num)
                    print(f"[CAM2] Interval {num}: {os.path.basename(fp)}")
                    return
            except OSError as e:
                if e.errno == 28:
                    print("[CAM2] â DISK FULL"); return
            except Exception as e:
                print(f"[CAM2] Save error (attempt {attempt}): {e}")
            time.sleep(0.5)

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  YOLO INFERENCE THREAD  (decoupled from main loop)
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _yolo_infer_thread(self):
        """
        Dedicated thread for serial.pt inference.
        Reads latest frame â runs detection â updates UI result.
        When serial detected AND scanning active â pushes crop to
        _ocr_crop_queue for immediate OCR processing (video_test_03_cv2.py style).

        FIX: Use frame-ID tracking instead of clearing _yolo_latest_frame=None.
        Clearing caused the main loop to see None and skip feeding the YOLO
        thread new frames, creating long gaps with no UI update. Now the
        thread checks _yolo_last_id vs _yolo_processed_id to detect new frames.
        
        FIX-10: Increased _YOLO_SKIP from 2 to 4 (every 4th frame) to reduce
        camera lag on Pi. With 30fps camera, this is 7.5 inferences/sec which
        is sufficient for real-time detection while keeping Pi responsive.
        """
        _YOLO_SKIP   = 2     # run YOLO on every 2nd frame
        _status_tick = 0     # for periodic status prints

        while self.running:
            # ── Periodic status print every ~5 s ─────────────────────────
            _status_tick += 1
            if _status_tick >= 100:
                _status_tick = 0
                print(f"[CAM2-YOLO] alive  "
                      f"_is_scanning={self._is_scanning}  "
                      f"frames={self._frame_count}  "
                      f"dets={self._yolo_det_count}  "
                      f"reader={'OK' if self.paddle_reader else 'loading'}",
                      flush=True)

            # ── ALWAYS run inference (removed _yolo_active gate) ──────────
            # Previously slept here if _yolo_active=False → nothing reached OCR.
            # Fix: run serial.pt on every frame; gate only queue push on _is_scanning.
            with self._yolo_frame_lock:
                cur_id = self._yolo_last_id
                frame  = self._yolo_latest_frame  # do NOT clear â main loop updates it

            if frame is None or cur_id == self._yolo_processed_id:
                time.sleep(0.02)
                continue

            # Skip frames to reduce CPU (run YOLO every _YOLO_SKIP frames received)
            if cur_id % _YOLO_SKIP != 0:
                self._yolo_processed_id = cur_id
                time.sleep(0.005)
                continue

            self._yolo_processed_id = cur_id

            try:
                crop, x1, y1, x2, y2, detected = \
                    self._run_yolo_on_frame(frame)



                # Update UI annotation result
                with self._yolo_result_lock:
                    self._yolo_latest_result = (crop, x1, y1, x2, y2, detected)

                # ââ Push crop to OCR queue (video_test_03_cv2.py style) ââ
                # Every detected frame goes immediately to OCR worker for
                # voting. Don't wait for 3 slots â process each frame now.
                if (detected
                        and crop is not None
                        and self._is_scanning
                        and not self.ocr_done):
                    elapsed = time.time() - self._scan_start_ts_ref[0]
                    try:
                        # Queue is unlimited â keep every crop for OCR voting.
                        # OCR confirms after 2 matches then skips the rest.
                        # Sharp crops only: skip blurry to save OCR time.
                        _c_sharp = float(
                            __import__('cv2').Laplacian(
                                __import__('cv2').cvtColor(crop, __import__('cv2').COLOR_BGR2GRAY)
                                if len(crop.shape) == 3 else crop,
                                0x06).var())  # CV_64F = 6
                        # Threshold 5.0 suits engraved metal serials (low Laplacian variance).
                        # Was 3.0 â raised slightly to skip clearly-blurry motion frames.
                        _SHARP_MIN = 5.0
                        if _c_sharp < _SHARP_MIN:
                            print(f'[CAM2-YOLO] Blurry crop skipped sharp={_c_sharp:.1f} '
                                  f'(min={_SHARP_MIN}) â waiting for sharper frame')
                        else:
                            self._yolo_det_count += 1
                            self._ocr_crop_queue.put_nowait(
                                (crop.copy(), frame.copy(),
                                 x1, y1, x2, y2, elapsed))
                            print(f'[CAM2-YOLO] â Queued crop #{self._yolo_det_count} '
                                  f'sharp={_c_sharp:.1f} '
                                  f'@ {elapsed:.2f}s  q={self._ocr_crop_queue.qsize()}')
                    except Exception as e:
                        print(f"[CAM2-YOLO] Queue push error: {e}")
                elif detected and crop is not None and not self._is_scanning:
                    print(f"[CAM2-YOLO] Detection ready but _is_scanning=False (waiting for start_ocr)")
                elif not detected:
                    if cur_id % 30 == 0:  # Log every 30 frames
                        print(f"[CAM2-YOLO] No detection @ frame {cur_id}")

            except Exception as e:
                print(f"[CAM2-YOLO-THREAD] error: {e}")

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  MAIN LOOP
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _main_loop(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|probesize;32|"
            "analyzeduration;0|fifo_size;50000")
        
        print(f"[CAM2-MAINLOOP] Starting frame reader for: {self.url[:80] if self.url else 'Not set'}")
        
        self._cap = (self.open_cap_fn(self.url)
                     if self.open_cap_fn
                     else _FFmpegCapture(self.url))
        
        print(f"[CAM2-MAINLOOP] â Capture device initialized, reading frames...")

        _last_good_ts   = None
        _RECONNECT_SECS = 12.0
        _scan_active    = False
        _scan_start_ts  = None
        _last_ocr_ts    = 0.0
        _frame_count    = 0

        while self.running:
            try:
                ok, frame = self._cap.read()
                
                # Log first successful frame
                if ok and frame is not None:
                    _frame_count += 1
                    if _frame_count == 1:
                        fh, fw = frame.shape[:2]
                        print(f"[CAM2-MAINLOOP] ð¹ First frame received! Resolution: {fw}Ã{fh}")

                # ââ Reconnect âââââââââââââââââââââââââââââââââ
                if not ok or frame is None or self._force_reconnect:
                    timeout = 3.0 if self._is_scanning else _RECONNECT_SECS
                    if (self._force_reconnect
                            or (_last_good_ts is not None
                                and time.time()-_last_good_ts > timeout)):
                        print("[CAM2-MAINLOOP] â ï¸ Reconnecting Camera-2...")
                        self._force_reconnect = False
                        try:
                            if hasattr(self._cap,'release'):
                                self._cap.release()
                        except Exception: pass
                        time.sleep(2.0)
                        self._cap = (self.open_cap_fn(self.url)
                                     if self.open_cap_fn
                                     else _FFmpegCapture(self.url))
                        _last_good_ts = None
                    time.sleep(0.1)
                    continue

                _last_good_ts = time.time()
                self._last_good_ts_global = _last_good_ts
                with self._lock:
                    self.last_frame = frame

                # Count frames and log first arrival
                self._frame_count += 1
                if self._frame_count == 1:
                    fh, fw = frame.shape[:2]
                    print(f"[CAM2-MAINLOOP] ð¹ First Camera-2 frame: {fw}Ã{fh}  "
                          f"serial.pt {'loaded â' if self.serial_detector is not None else 'loading...'}")

                # ââ FAST PATH: push raw frame to UI immediately ââââââ
                # _raw_frame is used by get_annotated_frame_with_det()
                # as a fallback when _latest_frame (set by YOLO thread)
                # is not yet available. This guarantees the stream
                # shows live video from the very first frame received,
                # with no waiting for YOLO to load or run.
                with self._raw_frame_lock:
                    self._raw_frame = frame

                # ââ Feed latest frame to YOLO thread (non-blocking) ââ
                # YOLO inference runs in _yolo_infer_thread â never blocks
                # the main loop, keeping Camera-2 stream smooth.
                if self._yolo_active:
                    with self._yolo_frame_lock:
                        self._yolo_latest_frame = frame
                        self._yolo_last_id += 1

                # ââ Read latest YOLO result (non-blocking) âââââ
                with self._yolo_result_lock:
                    crop, x1, y1, x2, y2, yolo_detected = \
                        self._yolo_latest_result

                # ââ Not scanning â skip slot/OCR logic ââââââââ
                if not self._is_scanning:
                    _scan_active = False
                    time.sleep(0.02)
                    continue

                # ââ Folder guard ââââââââââââââââââââââââââââââ
                if not self.panel_folder or self.panel_folder == ".":
                    self.status = "Waiting for folder..."
                    time.sleep(0.05)
                    continue

                # ââ New scan ââââââââââââââââââââââââââââââââââ
                if not _scan_active:
                    _scan_active   = True
                    _scan_start_ts = time.time()
                    _last_ocr_ts   = 0.0
                    print("[CAM2] ð Scan started â "
                          "waiting for serial.pt detections")

                elapsed = time.time() - _scan_start_ts

                # ââ Serial-appeared frames â MAIN folder (2 frames only) âââââ
                # Save exactly 2 frames to the panel root when YOLO first detects
                # the serial_number class.  No OCR_Frame / interval files any more.
                # Frame 1 = first detection moment
                # Frame 2 = sharpest detection frame seen so far
                if yolo_detected:
                    self._save_serial_appeared_frame(frame, crop, x1, y1, x2, y2)

                # ââ Progressive capture slots (YOLO ONLY - no ROI fallback) ââ
                # FIX-16: STRICT - only save crops from actual serial.pt detection
                # Remove ROI fallback from _save_slot_to_disk since serial_captures
                # must contain EXACT detection locations, not generic ROI regions
                if yolo_detected:
                    self._try_fill_slot(frame, crop,
                                        x1, y1, x2, y2, elapsed)
                    filled = sum(self._slots_saved)
                    self.status = (f"Capturing {filled}/"
                                   f"{self.BEST_FRAME_COUNT} frames [YOLO]...")
                    print(f"[CAM2] YOLO Detection â @ {elapsed:.1f}s â saving exact crop")

                elif not yolo_detected:
                    time_since_detect = elapsed - (self._last_yolo_detect_ts or 0.0)
                    self.status = f"Waiting for serial detection ({time_since_detect:.1f}s)..."

                # ââ Signal OCR worker once all 3 slots filled âââââââââ
                # The OCR worker runs in its own thread â the main loop
                # continues processing frames and running YOLO uninterrupted.
                # FIX-15: Guaranteed trigger even if slots filled quickly
                if (self._all_slots_filled()
                        and not self.ocr_done
                        and not self._ocr_running):
                    self._ocr_running = True
                    self.status = "OCR running in background..."
                    print(f"\n[CAM2] ð¯ ALL {self.BEST_FRAME_COUNT} SLOTS FILLED")
                    print(f"[CAM2] ð TRIGGERING PaddleOCR WORKER NOW")
                    filled = sum(self._slots_saved)
                    print(f"[CAM2]    Slots: {self._slots_saved}")
                    print(f"[CAM2]    Queue size: {self._ocr_crop_queue.qsize()}")
                    self._ocr_event.set()   # wake the OCR worker thread
                    print(f"[CAM2] â PaddleOCR signal sent\n")

                time.sleep(0.005)   # 200fps cap (was 0.02/50fps) â _raw_frame now
                                    # updates every 5ms â UI gets fresher frames

            except Exception as e:
                print(f"[CAM2] Loop error: {e}")
                time.sleep(0.5)

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  OCR WORKER THREAD
    #  Runs independently â never blocks main loop or YOLO inference
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _ocr_worker(self):
        """
        Per-frame OCR worker â exactly like video_test_03_cv2.py.

        Flow (matches video_test_03_cv2.py process_frame loop):
          1. Take crop from _ocr_crop_queue (put there by _yolo_infer_thread)
          2. Preprocess: 2Ã upscale â grey â 2DFilter + Otsu
          3. PaddleOCR on YOLO crop
          4. _correct_serial() â validate 9digit+1letter
          5. Vote: same serial STABLE_COUNT_REQ times = confirmed

        Also fills save slots for PDF frame captures.
        """
        _frame_count = 0

        while self.running:
            # Block until a crop arrives (or 0.5s timeout to check running)
            try:
                crop, frame, x1, y1, x2, y2, elapsed = \
                    self._ocr_crop_queue.get(timeout=0.5)
            except _queue_mod.Empty:
                continue

            # Skip only if serial already confirmed OR scanning explicitly stopped.
            # _yolo_active=False does NOT stop us â YOLO stopping just means no new
            # crops arrive, but we keep draining whatever is already in the queue.
            if self.ocr_done or not self._is_scanning:
                continue

            # ââ GATE: wait for PaddleOCR reader ââââââââââââââââââââââââ
            if self.paddle_reader is None:
                # Init permanently failed â log once then skip forever
                if getattr(self, '_paddle_init_failed', False):
                    if not getattr(self, '_easy_fail_logged', False):
                        print('[CAM2-OCR] ❌ PaddleOCR init failed — '
                              'check terminal for error above. '
                              'Crops will be discarded.', flush=True)
                        self._easy_fail_logged = True
                    continue

                # Init not finished yet â skip this crop but DON'T wait 20s
                if not getattr(self, '_paddle_init_done', False):
                    if _frame_count == 0:   # log once so it's visible
                        print('[CAM2-OCR] ⏳ PaddleOCR still loading '
                              'crop skipped (will retry next frame)', flush=True)
                    continue

                # Init finished but reader still None (shouldn't happen)
                print('[CAM2-OCR] ❌ PaddleOCR init done but reader=None — '
                      'crop skipped', flush=True)
                continue

            _frame_count += 1
            h, w = crop.shape[:2]
            print(f"[CAM2-OCR] â¶ frame#{_frame_count}  crop={w}Ã{h}  "
                  f"elapsed={elapsed:.2f}s", flush=True)

            serial = self._ocr_pipeline(crop)

            print(f"[CAM2-OCR] â frame#{_frame_count} â '{serial}'", flush=True)

            if serial:
                self.partial_serial = serial
                day    = serial[0:2]
                code   = serial[6:9]
                letter = serial[9]

                if self.day_confirmed    is None: self.day_votes[day]       += 1
                if self.code_confirmed   is None: self.code_votes[code]     += 1
                if self.letter_confirmed is None: self.letter_votes[letter] += 1

                d_cnt = self.day_votes[day]       if self.day_confirmed    is None else 0
                c_cnt = self.code_votes[code]     if self.code_confirmed   is None else 0
                l_cnt = self.letter_votes[letter] if self.letter_confirmed is None else 0

                CONFIRM_VOTES = 2
                print(f"[CAM2-OCR] vote day='{day}'({d_cnt}/{CONFIRM_VOTES}) "
                      f"code='{code}'({c_cnt}/{CONFIRM_VOTES}) "
                      f"letter='{letter}'({l_cnt}/{CONFIRM_VOTES})", flush=True)

                if d_cnt >= CONFIRM_VOTES and self.day_confirmed    is None:
                    self.day_confirmed = day
                    print(f"[CAM2-OCR] â­ DAY FROZEN: '{day}'", flush=True)
                if c_cnt >= CONFIRM_VOTES and self.code_confirmed   is None:
                    self.code_confirmed = code
                    print(f"[CAM2-OCR] â­ CODE FROZEN: '{code}'", flush=True)
                if l_cnt >= CONFIRM_VOTES and self.letter_confirmed is None:
                    self.letter_confirmed = letter
                    print(f"[CAM2-OCR] â­ LETTER FROZEN: '{letter}'", flush=True)

                self.status = (f"Day:{self.day_confirmed or day}  "
                               f"Code:{self.code_confirmed or code}  "
                               f"Letter:{self.letter_confirmed or letter}")

                if (self.day_confirmed is not None
                        and self.code_confirmed   is not None
                        and self.letter_confirmed is not None):
                    from datetime import datetime as _dtc
                    _n = _dtc.now()
                    final = (self.day_confirmed
                             + f"{_n.month:02d}{_n.year%100:02d}"
                             + self.code_confirmed + self.letter_confirmed)
                    print(f"[CAM2-OCR] âââ CONFIRMED: {final}", flush=True)
                    self._confirm_serial(final)
                    continue

            if not self._all_slots_filled():
                try:
                    self._try_fill_slot(frame, crop, x1, y1, x2, y2, elapsed)
                except Exception:
                    pass

    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    #  COMPAT
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

    def _basic_quality_ok(self, gray):
        if gray.std()<5 or gray.mean()<10 or gray.mean()>245:
            return False,"lighting"
        return True,"ok"

    def _background_audit(self, roi, serial, folder):
        try:
            log = [f"Audit: {serial}","="*20]
            for label,pimg in preprocess_engraved_metal(roi):
                raw_t, raw_s = self._run_paddle_ocr(pimg)
                log.append(f"[{label}] P={raw_t!r} score={raw_s:.3f}")
            with open(os.path.join(folder,"cam2_ocr_audit.txt"),
                      "a",encoding="utf-8") as f:
                f.write("\n".join(log)+"\n\n")
        except Exception: pass


if __name__ == "__main__":
    pass

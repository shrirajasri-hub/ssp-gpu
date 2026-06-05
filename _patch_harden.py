"""
Final hardening patch for camera2_ocr.py.
Uses direct line-number patching for lines with non-ASCII to avoid encoding issues.
"""
import sys

SRC = 'camera2_ocr.py'

with open(SRC, 'r', encoding='utf-8') as f:
    lines = f.readlines()

original_len = len(lines)
changes = []

def replace_lines(start_0idx, end_0idx, new_lines_text, label):
    """Replace lines[start_0idx:end_0idx] with new_lines_text (list of str)."""
    old = lines[start_0idx:end_0idx]
    print(f"\n[{label}] Replacing lines {start_0idx+1}-{end_0idx}:")
    for i, l in enumerate(old):
        print(f"  OLD L{start_0idx+1+i}: {repr(l)}")
    for i, l in enumerate(new_lines_text):
        print(f"  NEW   : {repr(l)}")
    lines[start_0idx:end_0idx] = new_lines_text
    changes.append(label)

# ─────────────────────────────────────────────────────────────────────────────
# Locate key lines by searching for unique markers
# ─────────────────────────────────────────────────────────────────────────────

def find_line(pattern, start=0):
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return i
    return -1

def find_block(patterns):
    """Find start index where all patterns appear consecutively."""
    for i in range(len(lines) - len(patterns)):
        if all(patterns[j] in lines[i+j] for j in range(len(patterns))):
            return i
    return -1

# ═══════════════════════════════════════════════════════════════════════════
# FIX 1d: Update _ocr_event declaration comment
# ═══════════════════════════════════════════════════════════════════════════
idx = find_line('# Solution: main loop signals this event; OCR worker runs independently.')
assert idx >= 0, "Could not find _ocr_event comment"
# Replace lines idx-2 to idx+3 (the comment block + declaration)
start = idx - 2
end   = idx + 4  # exclusive
replace_lines(start, end, [
    '        # -- OCR worker thread (separate from main loop) --\n',
    '        # The OCR worker is queue-driven: it blocks on\n',
    '        # _ocr_crop_queue.get(timeout=1) and never waits on _ocr_event.\n',
    '        # _ocr_event is retained ONLY so _init_paddle can signal\n',
    '        # PaddleOCR-ready status. All clear()/set() calls that gated\n',
    '        # the old EasyOCR worker have been removed by FIX 1.\n',
    '        # Do NOT add _ocr_event.wait() to the OCR worker.\n',
    '        self._ocr_event           = threading.Event()\n',
    '        self._ocr_running         = False   # True while OCR worker is active\n',
], "FIX 1d: _ocr_event comment")

# ═══════════════════════════════════════════════════════════════════════════
# FIX 1a: Remove _ocr_event.clear() from reset_for_new_panel (now around L911)
# ═══════════════════════════════════════════════════════════════════════════
idx = find_line('self._ocr_event.clear()', find_line('def reset_for_new_panel'))
assert idx >= 0, "Could not find event.clear() in reset_for_new_panel"
# Just this single line — replace with comment
replace_lines(idx, idx+1, [
    '            # FIX 1a: _ocr_event.clear() removed. Worker is queue-driven;\n',
    '            # it never calls _ocr_event.wait(). Clearing here was vestigial\n',
    '            # and could race against the PaddleOCR-ready .set() in _init_paddle.\n',
], "FIX 1a: event.clear() in reset_for_new_panel")

# ═══════════════════════════════════════════════════════════════════════════
# FIX 1b: Remove _ocr_event.clear() from start_ocr
# ═══════════════════════════════════════════════════════════════════════════
idx = find_line('self._ocr_event.clear()      # clear any pending OCR signal',
                find_line('def start_ocr'))
assert idx >= 0, "Could not find event.clear() in start_ocr"
replace_lines(idx, idx+1, [
    '            # FIX 1b: _ocr_event.clear() removed. Worker is queue-driven;\n',
    '            # clearing the event here was vestigial and had no effect.\n',
], "FIX 1b: event.clear() in start_ocr")

# ═══════════════════════════════════════════════════════════════════════════
# FIX 1c: Remove _ocr_event.set() from main loop slot-fill branch
# ═══════════════════════════════════════════════════════════════════════════
idx = find_line('self._ocr_event.set()   # wake the OCR worker thread')
assert idx >= 0, "Could not find _ocr_event.set() in main loop"
# Also remove the print below it
next_is_print = 'EasyOCR signal sent' in lines[idx+1]
end = idx + 2 if next_is_print else idx + 1
replace_lines(idx, end, [
    '                    # FIX 1c: _ocr_event.set() removed. Worker is queue-driven.\n',
    '                    # Crop was already placed in _ocr_crop_queue by\n',
    '                    # _yolo_infer_thread; no event signal needed.\n',
    f'                    print(f"[CAM2] OCR queue size after slot fill: {{self._ocr_crop_queue.qsize()}}")\n',
], "FIX 1c: event.set() in main loop")

# ═══════════════════════════════════════════════════════════════════════════
# FIX 2: Replace _confirm_serial folder fallback
# ═══════════════════════════════════════════════════════════════════════════
idx_confirm_start = find_line('def _confirm_serial')
assert idx_confirm_start >= 0

# Find the fallback block start: "# CHANGE 6b"
idx = find_line('# CHANGE 6b: use capture-time folder, not self.panel_folder',
                idx_confirm_start)
assert idx >= 0, "Could not find CHANGE 6b comment"

# Find end of the block: look for the blank line after the except block
end = idx + 1
# We need to find the end: it ends after "except Exception as _e: print(..."
# Scan forward for the blank line after the except block
depth = 0
for j in range(idx, min(idx+30, len(lines))):
    if j > idx and lines[j].strip() == '':
        end = j
        break
else:
    end = idx + 18  # fallback

print(f"\nFIX 2 block: lines {idx+1} to {end}")
for i in range(idx, end):
    print(f"  L{i+1}: {repr(lines[i])}")

replace_lines(idx, end, [
    '        # FIX 2: Never fall back to self.panel_folder.\n',
    '        # panel_folder is the capture-time folder from the queue item and is\n',
    '        # the ONLY safe source. If it is missing/invalid, we refuse to write\n',
    '        # rather than risk writing Panel A results into Panel B folder.\n',
    '        if not panel_folder or panel_folder == ".":\n',
    '            print(\n',
    '                "[OCR] No valid capture-time folder supplied. "\n',
    '                "Refusing to write OCR result to disk.",\n',
    '                flush=True\n',
    '            )\n',
    '        else:\n',
    '            folder = panel_folder\n',
    '            print(f"[OCR] Saving serial to {folder} | session={session_id}",\n',
    '                  flush=True)   # FIX 5: saving log\n',
    '            try:\n',
    '                os.makedirs(folder, exist_ok=True)\n',
    '                with open(os.path.join(folder, "serial_ocr_result.txt"),\n',
    '                          "w", encoding="utf-8") as _rf:\n',
    '                    _rf.write(f"Serial:    {serial}\\n"\n',
    '                              f"Timestamp: {datetime.now().isoformat()}\\n"\n',
    '                              f"Session:   {session_id}\\n"\n',
    '                              f"Method:    serial.pt + PaddleOCR\\n")\n',
    '                print(f"[OCR] serial_ocr_result.txt written to "\n',
    '                      f"{os.path.basename(folder)}", flush=True)\n',
    '            except Exception as _e:\n',
    '                print(f"[CAM2] result write error: {_e}", flush=True)\n',
    '\n',
], "FIX 2: _confirm_serial folder fallback removed")

# ═══════════════════════════════════════════════════════════════════════════
# FIX 3: Session re-validation before callback
# ═══════════════════════════════════════════════════════════════════════════
idx = find_line('if self.on_serial_detected:', idx_confirm_start)
assert idx >= 0, "Could not find on_serial_detected"
# The block is 3 lines: if / try: self.on_serial_detected / except
end = idx + 3
print(f"\nFIX 3 block: lines {idx+1} to {end}")
for i in range(idx, end):
    print(f"  L{i+1}: {repr(lines[i])}")

replace_lines(idx, end, [
    '        # FIX 3: re-validate session before firing the callback.\n',
    '        # Between the lock release above and here, reset_for_new_panel()\n',
    '        # may have incremented _panel_session_id. Firing the callback with\n',
    '        # stale session data would update Panel B state with Panel A results.\n',
    '        with self._lock:\n',
    '            _cb_ok = (session_id is None or\n',
    '                      session_id == self._panel_session_id)\n',
    '        if not _cb_ok:\n',
    '            print(\n',
    '                f"[CAM2] Callback skipped — stale session "\n',
    '                f"(item={session_id} current={self._panel_session_id})",\n',
    '                flush=True\n',
    '            )\n',
    '        elif self.on_serial_detected:\n',
    '            try:\n',
    '                self.on_serial_detected(serial)\n',
    '            except Exception as e:\n',
    '                print(f"[CAM2] Callback error: {e}")\n',
], "FIX 3: session re-validation before callback")

# ═══════════════════════════════════════════════════════════════════════════
# FIX 5a: OCR worker startup diagnostics
# ═══════════════════════════════════════════════════════════════════════════
idx = find_line('"[CAM2-OCR] OCR Worker Started"', find_line('def _ocr_worker'))
assert idx >= 0, "Could not find OCR Worker Started print"
# Replace single line
replace_lines(idx, idx+1, [
    '        print(\n',
    '            f"[OCR] Worker started | "\n',
    '            f"session={self._panel_session_id} | "\n',
    '            f"queue_size={self._ocr_crop_queue.qsize()}",\n',
    '            flush=True\n',
    '        )   # FIX 5a: startup diagnostic\n',
], "FIX 5a: worker startup diagnostic")

# ═══════════════════════════════════════════════════════════════════════════
# FIX 5b: Per-item processing diagnostic (replace verbose block)
# ═══════════════════════════════════════════════════════════════════════════
idx = find_line('# TASK 1: detailed session / folder log', find_line('def _ocr_worker'))
assert idx >= 0, "Could not find TASK 1 detailed session log"
end = idx + 6  # 6 lines: comment + 5 prints
print(f"\nFIX 5b block: lines {idx+1} to {end}")
for i in range(idx, end):
    print(f"  L{i+1}: {repr(lines[i])}")

replace_lines(idx, end, [
    '                        # FIX 5b: compact per-item diagnostic\n',
    '                        print(\n',
    '                            f"[OCR] Processing crop | "\n',
    '                            f"session={item_session} | "\n',
    '                            f"panel={item_panel_id}",\n',
    '                            flush=True\n',
    '                        )\n',
    '                        print(f"[OCR] Queue Session   = {item_session}", flush=True)\n',
    '                        print(f"[OCR] Current Session = {cur_session}",  flush=True)\n',
    '                        print(f"[OCR] Panel Folder    = {item_folder}",  flush=True)\n',
], "FIX 5b: per-item diagnostic")

# ═══════════════════════════════════════════════════════════════════════════
# Write file
# ═══════════════════════════════════════════════════════════════════════════

with open(SRC, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\n{'='*60}")
print(f"Patched {SRC}  ({original_len} -> {len(lines)} lines)")
print(f"Changes applied: {len(changes)}")
for i, c in enumerate(changes, 1):
    print(f"  [{i}] {c}")

# ═══════════════════════════════════════════════════════════════════════════
# Post-patch checks
# ═══════════════════════════════════════════════════════════════════════════
print("\nPost-patch checks:")
with open(SRC, 'r', encoding='utf-8') as f:
    final = f.read()
    flines = final.splitlines()

clears  = [i+1 for i, l in enumerate(flines) if '_ocr_event.clear()' in l]
sets    = [i+1 for i, l in enumerate(flines) if '_ocr_event.set()' in l]
pf_conf = []
in_c = False
for i, l in enumerate(flines, 1):
    if 'def _confirm_serial' in l: in_c = True
    if in_c and i > 1960 and l.strip().startswith('def ') and '_confirm_serial' not in l:
        in_c = False
    if in_c and 'self.panel_folder' in l and 'def _confirm_serial' not in l:
        pf_conf.append((i, l.strip()))

print(f"  _ocr_event.clear() calls: {clears} (expected: [])")
print(f"  _ocr_event.set()   calls: {sets}   (expected: [L778, L797] only)")
print(f"  self.panel_folder in _confirm_serial: {pf_conf} (expected: [])")

import os
import glob
import traceback
import cv2
import gc
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(serial_number, panel_folder, seq_times=None, total_time=0, panel_start=None, cam2_image=None):
    """Generates a professional 6-page inspection report for the industrial panel."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_name = f"{serial_number}_Inspection_Report.pdf"
    report_path = os.path.join(panel_folder, report_name)
    
    print(f"\n📄 GENERATING PDF: {report_name}")
    
    doc = SimpleDocTemplate(report_path, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'],
                                 fontSize=22, spaceAfter=20, alignment=1, textColor=colors.HexColor("#1a237e"))
    h1_style    = ParagraphStyle('H1Style', parent=styles['Heading1'],
                                 fontSize=16, spaceAfter=12, textColor=colors.HexColor("#1a237e"))
    h2_style    = ParagraphStyle('H2Style', parent=styles['Heading2'],
                                 fontSize=14, spaceAfter=10, textColor=colors.HexColor("#1565c0"))
    sub_style   = ParagraphStyle('SubStyle', parent=styles['Normal'],
                                 fontSize=10, leading=14, spaceAfter=8)
    
    USABLE_W = A4[0] - 80

    def _find(folder, *patterns):
        """Try multiple glob patterns in order; return first non-empty match.
        FIX 3: Search both folder root AND one level of subdirectories for
        robustness when the serial-rename step produced a nested path."""
        import glob as _glob
        for pat in patterns:
            # Direct match in the given folder
            results = sorted(_glob.glob(os.path.join(folder, pat)))
            if results:
                return results
            # One-level deep (e.g. SSP-SEQ/2025-01-01/SSP-SEQ_123456/<file>)
            results = sorted(_glob.glob(os.path.join(folder, '**', pat)))
            if results:
                return results
        return []

    def _find_best(folder, *patterns):
        """Like _find but returns the sharpest image (highest Laplacian variance)
        from all matches — important when multiple frames were saved per sequence."""
        import glob as _glob
        import cv2 as _cv2
        seen = set()
        candidates = []
        for pat in patterns:
            for p in _glob.glob(os.path.join(folder, pat)):
                rp = os.path.realpath(p)
                if rp not in seen: seen.add(rp); candidates.append(p)
            for p in _glob.glob(os.path.join(folder, '**', pat)):
                rp = os.path.realpath(p)
                if rp not in seen: seen.add(rp); candidates.append(p)
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates
        # Pick sharpest
        best_path  = candidates[0]
        best_var   = -1.0
        for p in candidates:
            try:
                img = _cv2.imread(p)
                if img is None:
                    continue
                gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
                var  = _cv2.Laplacian(gray, _cv2.CV_64F).var()
                if var > best_var:
                    best_var  = var
                    best_path = p
            except Exception:
                pass
        return [best_path]

    def _fit_image(img_path, max_w_pt, max_h_pt):
        try:
            img = Image(img_path)
            iw, ih = img.imageWidth, img.imageHeight
            aspect = ih / iw
            
            w = min(iw, max_w_pt)
            h = w * aspect
            
            if h > max_h_pt:
                h = max_h_pt
                w = h / aspect
                
            img.drawWidth = w
            img.drawHeight = h
            return img
        except Exception as e:
            return Paragraph(f"Error loading image: {e}", sub_style)

    # ── PAGE 1: COVER & SUMMARY ──────────────────────────────
    story.append(Paragraph("Industrial Panel Inspection Report", title_style))
    story.append(Spacer(1, 20))
    
    meta_data = [
        ["Serial Number:", serial_number],
        ["Inspection Date:", timestamp],
        ["Status:", "PASS" if serial_number != "UNKNOWN" else "MANUAL REVIEW REQUIRED"],
        ["Location:", "Line 01 - Final Assembly"]
    ]
    t_meta = Table(meta_data, colWidths=[1.5*inch, 4*inch])
    t_meta.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 30))
    
    # Lead time table
    if seq_times:
        story.append(Paragraph("Sequence Lead Times", h1_style))
        lt_rows = [["Sequence Step", "Time Taken", "Status"]]
        seq_names = {1: "SEQ 1: Top Surface", 2: "SEQ 2: Middle Region", 3: "SEQ 3: Bottom Region"}
        for i in (1, 2, 3):
            t = seq_times.get(i)
            if t:
                m, s = divmod(int(t), 60)
                t_str = f"{m}m {s}s" if m else f"{s}s"
                status = "✓ Completed"
            else:
                t_str = "—"
                status = "✕ Missed"
            lt_rows.append([seq_names[i], t_str, status])
        
        t_lt = Table(lt_rows, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        t_lt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a237e")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t_lt)
        
    story.append(Spacer(1, 20))
    m, s = divmod(int(total_time), 60)
    tt_str = f"{m}m {s}s" if m else f"{s}s"
    story.append(Paragraph(f"<b>Total Inspection Lead Time:</b> {tt_str}", sub_style))
    story.append(PageBreak())

    # ── PAGE 2: SEQ 1 FULL ───────────────────────────────────
    story.append(Paragraph("Sequence 1 — Full Panel Image", h1_style))
    files = _find_best(panel_folder, "*_Seq1_Full_*.jpg", "SSP-SEQ_Seq1_Full_*.jpg", "*Seq1*Full*.jpg")
    if files:
        story.append(_fit_image(files[0], USABLE_W, 7*inch))
        print(f"  [PDF] SEQ1: {os.path.basename(files[0])}")
    else:
        story.append(Paragraph("<i>SEQ1 full image not captured.</i>", sub_style))
        print("  [PDF] WARNING: SEQ1 full image MISSING")
    story.append(PageBreak())

    # ── PAGE 3: SEQ 1 CROPS (Left, Middle, Right, Bottom) ────
    story.append(Paragraph("Sequence 1 — Zone Crops", h1_style))
    story.append(Spacer(1, 10))
    
    crop_names = ['Left', 'Middle', 'Right', 'Bottom']
    crop_images = []
    
    for cname in crop_names:
        # FIX 3: search crops subfolder first, then root folder, then any depth
        cfiles = _find_best(os.path.join(panel_folder, 'crops'),
                       f"*_Seq1_{cname}_*.jpg", f"*_Seq1_{cname.lower()}_*.jpg",
                       f"SSP-SEQ_Seq1_{cname}_*.jpg")
        if not cfiles:
            cfiles = _find_best(panel_folder,
                           f"*_Seq1_{cname}_*.jpg", f"*_Seq1_{cname.lower()}_*.jpg")
        if cfiles:
            crop_images.append((cname, cfiles[0]))
        else:
            print(f"  [PDF] WARNING: SEQ1 {cname} crop MISSING")

    if crop_images:
        half_w = (USABLE_W / 2) - 10
        grid = []
        for i in range(0, len(crop_images), 2):
            row_labels = []
            row_imgs = []
            
            # First item in row
            cname1, cfile1 = crop_images[i]
            row_labels.append(Paragraph(f"<b>{cname1.capitalize()} Zone</b>", sub_style))
            row_imgs.append(_fit_image(cfile1, half_w, 2.5*inch))
            
            # Second item in row (if exists)
            if i + 1 < len(crop_images):
                cname2, cfile2 = crop_images[i+1]
                row_labels.append(Paragraph(f"<b>{cname2.capitalize()} Zone</b>", sub_style))
                row_imgs.append(_fit_image(cfile2, half_w, 2.5*inch))
            else:
                row_labels.append(Spacer(1, 1))
                row_imgs.append(Spacer(1, 1))
                
            grid.append(row_labels)
            grid.append(row_imgs)
            grid.append([Spacer(1, 10), Spacer(1, 10)])
            
        story.append(Table(grid, colWidths=[half_w+5, half_w+5]))
        
    story.append(PageBreak())

    # ── PAGE 4: SEQ 2 FULL ───────────────────────────────────
    story.append(Paragraph("Sequence 2 — Full Panel Image", h1_style))
    files = _find_best(panel_folder, "*_Seq2_Full_*.jpg", "SSP-SEQ_Seq2_Full_*.jpg", "*Seq2*Full*.jpg")
    if files:
        story.append(_fit_image(files[0], USABLE_W, 7*inch))
        print(f"  [PDF] SEQ2: {os.path.basename(files[0])}")
    else:
        story.append(Paragraph("<i>SEQ2 full image not captured.</i>", sub_style))
        print("  [PDF] WARNING: SEQ2 full image MISSING")
    story.append(PageBreak())

    # ── PAGE 5: SEQ 3 FULL ───────────────────────────────────
    story.append(Paragraph("Sequence 3 — Full Panel Image", h1_style))
    files = _find_best(panel_folder, "*_Seq3_Full_*.jpg", "SSP-SEQ_Seq3_Full_*.jpg", "*Seq3*Full*.jpg")
    if files:
        story.append(_fit_image(files[0], USABLE_W, 7*inch))
        print(f"  [PDF] SEQ3: {os.path.basename(files[0])}")
    else:
        story.append(Paragraph("<i>SEQ3 full image not captured.</i>", sub_style))
        print("  [PDF] WARNING: SEQ3 full image MISSING")
    story.append(PageBreak())

    # ── PAGE 6: SERIAL NUMBER CAPTURE ────────────────────────
    story.append(Paragraph("6. Serial Number — Camera 2 Capture", h1_style))
    story.append(Spacer(1, 15))

    import glob as _gl

    # ── 6a. Serial-captures section (new file structure) ─────────────────────
    # camera2_ocr_v2.py saves to serial_captures/:
    #   CAM2_ROI_Annotated_HHMMSS.jpg  — full frame with green detection bbox
    #   CAM2_Full_Frame_HHMMSS.jpg     — clean full frame (slot 1)
    #   CAM2_Crop_HHMMSS.jpg           — exact HEF detection region
    #   CAM2_Crop_2_HHMMSS.jpg         — second crop for voting
    #   CAM2_Best_Full.jpg             — sharpest full frame across all slots
    # Main folder also has:
    #   CAM2_Serial_Appeared_1_*.jpg   — first frame when serial class appeared
    #   CAM2_Serial_Appeared_Best_*.jpg — sharpest serial-appeared frame
    serial_cap_dir = os.path.join(panel_folder, "serial_captures")

    _best_full   = os.path.join(serial_cap_dir, "CAM2_Best_Full.jpg")
    _roi_frames  = (
        sorted(_gl.glob(os.path.join(serial_cap_dir, "CAM2_ROI_Annotated_*.jpg"))) or
        sorted(_gl.glob(os.path.join(panel_folder, "**", "CAM2_ROI_Annotated_*.jpg")))
    )
    _full_frames = (
        sorted(_gl.glob(os.path.join(serial_cap_dir, "CAM2_Full_Frame_*.jpg"))) or
        sorted(_gl.glob(os.path.join(panel_folder, "**", "CAM2_Full_Frame_*.jpg")))
    )
    _crop_frames = (
        sorted(_gl.glob(os.path.join(serial_cap_dir, "CAM2_Crop_*.jpg"))) or
        sorted(_gl.glob(os.path.join(panel_folder, "**", "CAM2_Crop_*.jpg")))
    )
    # serial-appeared frames in main folder
    _appeared_frames = sorted(
        _gl.glob(os.path.join(panel_folder, "CAM2_Serial_Appeared_*.jpg"))
    )

    # Resolve best full frame: CAM2_Best_Full.jpg first, then slot full, then appeared
    _best_full_path = (
        _best_full         if os.path.exists(_best_full) else
        (_full_frames[0]   if _full_frames else None)   or
        (_appeared_frames[0] if _appeared_frames else None)
    )

    print(f"  [PDF] serial_captures/ → roi={len(_roi_frames)}  "
          f"full={len(_full_frames)}  crop={len(_crop_frames)}  "
          f"appeared={len(_appeared_frames)}")

    # Backwards-compat: also accept old file names from previous camera2_ocr.py
    if not _roi_frames and not _full_frames and not _crop_frames:
        _full_frames = (
            sorted(_gl.glob(os.path.join(serial_cap_dir, "CAM2_Raw_Frame_*.jpg"))) or
            sorted(_gl.glob(os.path.join(panel_folder, "**", "CAM2_Raw_Frame_*.jpg")))
        )
        _crop_frames = (
            sorted(_gl.glob(os.path.join(serial_cap_dir, "CAM2_Serial_Crop_*.jpg"))) or
            sorted(_gl.glob(os.path.join(panel_folder, "**", "CAM2_Serial_Crop_*.jpg")))
        )
        _zoom_frames = (
            sorted(_gl.glob(os.path.join(serial_cap_dir, "CAM2_Serial_Zoom_*.jpg"))) or
            sorted(_gl.glob(os.path.join(panel_folder, "**", "CAM2_Serial_Zoom_*.jpg")))
        )
        if _full_frames: _best_full_path = _full_frames[0]

    if _best_full_path or _roi_frames or _crop_frames:
        story.append(Paragraph("Serial Number — Camera 2 Detection", h2_style))
        story.append(Spacer(1, 8))

        # Row 1: ROI annotated (left) + Best full frame (right)
        half_w = (USABLE_W / 2) - 6
        row_imgs, row_labels = [], []

        if _roi_frames and os.path.exists(_roi_frames[0]):
            row_labels.append(Paragraph("<b>Detection region (YOLO bbox)</b>", sub_style))
            row_imgs.append(_fit_image(_roi_frames[0], half_w, 3.5*inch))
            print(f"  [PDF] ROI annotated → {os.path.basename(_roi_frames[0])}")
        elif _appeared_frames:
            row_labels.append(Paragraph("<b>Serial class appeared</b>", sub_style))
            row_imgs.append(_fit_image(_appeared_frames[0], half_w, 3.5*inch))

        if _best_full_path and os.path.exists(_best_full_path):
            row_labels.append(Paragraph("<b>Best full frame (sharpest)</b>", sub_style))
            row_imgs.append(_fit_image(_best_full_path, half_w, 3.5*inch))
            print(f"  [PDF] Best full → {os.path.basename(_best_full_path)}")

        if row_imgs:
            while len(row_imgs) < 2:
                row_labels.append(Spacer(1, 1))
                row_imgs.append(Spacer(1, 1))
            story.append(Table([row_labels, row_imgs],
                               colWidths=[half_w + 6, half_w + 6]))
            story.append(Spacer(1, 14))

        # Row 2: Crop (detection region only)
        if _crop_frames and os.path.exists(_crop_frames[0]):
            story.append(Paragraph("Serial Number — Crop (exact detection region)", h2_style))
            story.append(Spacer(1, 8))
            story.append(_fit_image(_crop_frames[0], USABLE_W * 0.5, 2.5*inch))
            story.append(Spacer(1, 14))
            print(f"  [PDF] Crop → {os.path.basename(_crop_frames[0])}")

        story.append(Paragraph(
            f"<b>Confirmed Serial Number:</b>  {serial_number}",
            ParagraphStyle('SerialLabel', parent=sub_style,
                           fontSize=13, textColor=colors.HexColor("#1a237e"))))
        story.append(Spacer(1, 20))

    else:
        story.append(Paragraph("Serial Number — Camera 2 Capture", h2_style))
        story.append(Spacer(1, 8))
        story.append(Paragraph("<i>No serial capture frames found.</i>", sub_style))
        story.append(Spacer(1, 12))

    # ── 6b. CAM2_Progress milestones — skipped (no longer saved) ────────────
    # Block removed to fix 'img_path' not defined error

    # ── 6c. Legacy images (if present) ───────────────────────────────────────
    def _find_serial(folder, sn):
        def _f(pat):
            return (_find(folder, f"{sn}_{pat}") or
                    _find(folder, f"SSP-SEQ_{pat}") or
                    _find(folder, f"Attempt_{pat}") or
                    _find(folder, pat))
        orig  = _f("CLEAN.jpg") or _f("Serial_*_Original_*.jpg") or _f("Serial_Original_*.jpg")
        rot   = _f("Serial_PDF_180_*.jpg") or _f("Serial_*_R180_*.jpg") or _f("Serial_Rotated180_*.jpg")
        zooms = _f("Cam2_OCR_Zoomed.jpg") or _f("Serial_*_Zoomed_*.jpg") or _find(folder, "*_Cam2_ROI_Zoomed_*.jpg")
        return orig, rot, zooms

    orig_files, rot_files, zoom_files = _find_serial(panel_folder, serial_number)

    half_w = (USABLE_W / 2) - 10
    if orig_files or rot_files:
        tbl_data = [
            [Paragraph("<b>Original View (0°)</b>", sub_style),
             Paragraph("<b>OCR View (180°)</b>", sub_style)],
            [
                _fit_image(orig_files[0], half_w, 2.5*inch) if orig_files else Paragraph("N/A", sub_style),
                _fit_image(rot_files[0],  half_w, 2.5*inch) if rot_files  else Paragraph("N/A", sub_style)
            ]
        ]
        story.append(Table(tbl_data, colWidths=[half_w+5, half_w+5]))
        story.append(Spacer(1, 20))

    # Final build
    try:
        doc.build(story)
        print(f"✅ PDF successfully generated: {report_path}")
        gc.collect()  # Free memory
        return report_path
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        traceback.print_exc()
        return None

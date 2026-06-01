import sys, os, traceback
sys.path.insert(0, r'C:\Users\shrir\Downloads\finalseq-ssp\GPU_SSP')
from camera2_ocr import SerialYOLODetector
model_path = r'C:\Users\shrir\Downloads\finalseq-ssp\GPU_SSP\models\serial.pt'
print('MODEL PATH:', model_path, 'exists=', os.path.exists(model_path))
try:
    d = SerialYOLODetector(model_path)
    print('SerialYOLODetector loaded. device=', getattr(d,'_device',None))
    print('Running quick detect on blank frame...')
    import numpy as np
    res = d.detect(np.zeros((960,1280,3), dtype='uint8'))
    print('detect() returned:', res)
except Exception as e:
    print('ERROR loading/initializing SerialYOLODetector:')
    traceback.print_exc()

# Seven Segment OCR (Pure Python) – Home Assistant Custom Integration (HACS)

Diese Version hat **keine OpenCV-Pip-Abhängigkeit**. Hintergrund: In manchen HA-Umgebungen (z.B. Python 3.13)
scheitert `opencv-python-headless` (keine Wheels → Build-from-source → Toolchain/Setuptools/Wheel-Probleme).

## Features
- Config Flow (UI-Setup)
- Crop + Rotate
- Vorverarbeitung (Pillow + NumPy): Autokontrast, Gaussian Blur, adaptives Threshold (Mean), Border-Clear, Despeckle
- 7‑Segment-Erkennung (Segment-ROI‑Checks)

## Installation (HACS Custom Repository)
1. Repository in HACS als **Integration** hinzufügen (Custom repository).
2. Installieren
3. Home Assistant neu starten
4. **Einstellungen → Geräte & Dienste → Integration hinzufügen → “Seven Segment OCR (Pure Python)”**

## Lizenz
MIT

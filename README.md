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


## Kalibrierung (Crop & Parameter)
Home Assistant Config Flows können **kein** interaktives Crop-Rechteck auf einem Kamera-Snapshot anbieten.
Empfehlung:
1. Nutze die Web-UI (dein vorheriges Projekt), um Crop & Preprocess zu finden.
2. Kopiere die Werte und trage sie im Options-Dialog dieser Integration ein.
Optional (v0.2.1): Du kannst im Options-Dialog auch ein JSON-Preset einfügen (siehe unten).


## Last-known-value (Kamera-Aussetzer)
Wenn die Kamera/der MJPEG-Stream kurz ausfällt, behält der Sensor den **letzten gültigen Wert**.
Im Attribut `image_error` siehst du dann den Grund (z.B. Timeout).


## HACS Icon
Im Repo liegen `icon.png` und `logo.png`, damit HACS ein hübsches Icon anzeigen kann.

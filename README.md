# Seven Segment OCR (OpenCV) – Home Assistant Custom Integration (HACS)

Diese Integration liest 7-Segment-Anzeigen aus einem **camera.\*** Entity – mit **OpenCV Vorverarbeitung** (CLAHE + adaptive Threshold) und einer **rein-Python 7-Segment-Erkennung** (kein `ssocr` Binary nötig).

## Features
- UI-Setup via Config Flow
- Crop/Rotate
- OpenCV Preprocessing: CLAHE, Blur, Adaptive Threshold, Border Clear, Despeckle
- Sensor mit erkannter Zahl als `state`, plus Debug-Attribute

## Installation (HACS Custom Repository)
1. Repository in HACS als **Integration** hinzufügen (Custom repository). citeturn0search9
2. Installieren
3. Home Assistant neu starten
4. **Einstellungen → Geräte & Dienste → Integration hinzufügen → “Seven Segment OCR (OpenCV)”**

## Hinweise
- Die Integration nutzt `opencv-python-headless`. Auf x86_64 ist das in der Regel problemlos, auf exotischen Architekturen kann es sein, dass keine Wheels verfügbar sind.

## Debug
Im Sensor findest du Attribute wie:
- `digits` (Liste)
- `segments` (pro Digit)
- `boxes` (Bounding Boxes)
- `preprocess` (aktive Preprocessing-Parameter)

## Lizenz
MIT

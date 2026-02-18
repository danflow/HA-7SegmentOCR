from __future__ import annotations

from datetime import timedelta
import logging
from typing import Optional, Tuple, List, Dict, Any, Deque
from collections import deque
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, ImageFilter

from homeassistant.components.camera import async_get_image
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.helpers.entity import Entity

from .const import (
    DOMAIN, DEFAULT_NAME,
    CONF_CAMERA, CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL,
    CONF_CROP_X, CONF_CROP_Y, CONF_CROP_W, CONF_CROP_H, CONF_ROTATE,
    CONF_EXPECTED_DIGITS,
    CONF_AUTOCONTRAST, CONF_BLUR, CONF_BLOCK_SIZE, CONF_C,
    CONF_BORDER_CLEAR, CONF_MIN_AREA, CONF_FORCE_INVERT,
    CONF_ALLOW_DECREASE, CONF_OUTPUT_TYPE, DEFAULT_OUTPUT_TYPE, DEFAULT_ALLOW_DECREASE,
)

_LOGGER = logging.getLogger(__name__)

SEGMENT_MAP = {
    (1,1,1,1,1,1,0): "0",
    (0,1,1,0,0,0,0): "1",
    (1,1,0,1,1,0,1): "2",
    (1,1,1,1,0,0,1): "3",
    (0,1,1,0,0,1,1): "4",
    (1,0,1,1,0,1,1): "5",
    (1,0,1,1,1,1,1): "6",
    (1,1,1,0,0,0,0): "7",
    (1,1,1,1,1,1,1): "8",
    (1,1,1,1,0,1,1): "9",
}

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _parse_value(raw: str, output_type: str):
    raw = (raw or "").strip()
    if output_type == "int":
        return int(raw), raw
    if output_type == "float":
        return float(raw.replace(",", ".")), raw
    return raw, raw

def _rotate_pil(im: Image.Image, rotate: int) -> Image.Image:
    r = int(rotate) % 360
    if r == 90:
        return im.transpose(Image.ROTATE_270)  # CW 90
    if r == 180:
        return im.transpose(Image.ROTATE_180)
    if r == 270:
        return im.transpose(Image.ROTATE_90)   # CW 270
    return im

def adaptive_threshold_mean(gray: np.ndarray, block: int, c: int) -> np.ndarray:
    """Adaptive mean threshold using integral image. Output 0/255 (black/white)."""
    h, w = gray.shape
    block = _clamp(int(block), 3, 199)
    if block % 2 == 0:
        block += 1
    r = block // 2

    padded = np.pad(gray.astype(np.uint32), ((1,1),(1,1)), mode="edge")
    ii = padded.cumsum(axis=0).cumsum(axis=1)

    y = np.arange(h)
    x = np.arange(w)
    y1 = np.clip(y - r, 0, h-1) + 1
    y2 = np.clip(y + r, 0, h-1) + 1
    x1 = np.clip(x - r, 0, w-1) + 1
    x2 = np.clip(x + r, 0, w-1) + 1

    A = ii[y1[:,None]-1, x1[None,:]-1]
    B = ii[y1[:,None]-1, x2[None,:]]
    C = ii[y2[:,None], x1[None,:]-1]
    D = ii[y2[:,None], x2[None,:]]

    area = (y2 - y1 + 1)[:,None] * (x2 - x1 + 1)[None,:]
    mean = (D - B - C + A) / area

    thresh = mean.astype(np.int32) - int(c)
    fg = gray.astype(np.int32) < thresh
    return np.where(fg, 0, 255).astype(np.uint8)

def normalize_polarity(bin_img: np.ndarray, force_invert: bool) -> np.ndarray:
    # prefer white background
    if float(bin_img.mean()) < 127:
        bin_img = 255 - bin_img
    if force_invert:
        bin_img = 255 - bin_img
    return bin_img

def clear_border(bin_img: np.ndarray, px: int) -> np.ndarray:
    px = _clamp(int(px), 0, 50)
    if px <= 0:
        return bin_img
    out = bin_img.copy()
    out[:px,:] = 255
    out[-px:,:] = 255
    out[:,:px] = 255
    out[:,-px:] = 255
    return out

def despeckle(bin_img: np.ndarray, min_area: int) -> np.ndarray:
    """Remove small black components using BFS (8-neighborhood)."""
    min_area = _clamp(int(min_area), 0, 1000000)
    if min_area <= 0:
        return bin_img

    h, w = bin_img.shape
    fg = (bin_img == 0)
    visited = np.zeros((h,w), dtype=bool)
    out = bin_img.copy()

    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for y0 in range(h):
        xs = np.where(fg[y0] & ~visited[y0])[0]
        for x0 in xs:
            if visited[y0, x0]:
                continue
            q: Deque[Tuple[int,int]] = deque()
            q.append((y0, x0))
            visited[y0, x0] = True
            comp = [(y0, x0)]
            while q:
                y, x = q.popleft()
                for dy, dx in neigh:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
                        comp.append((ny, nx))
            if len(comp) < min_area:
                for yy, xx in comp:
                    out[yy, xx] = 255
    return out

def find_digit_boxes(bin_img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    h, w = bin_img.shape
    col_has_fg = (bin_img == 0).any(axis=0)
    boxes_x = []
    in_run = False
    start = 0
    for x in range(w):
        if col_has_fg[x] and not in_run:
            in_run = True
            start = x
        elif not col_has_fg[x] and in_run:
            end = x-1
            if end - start >= 5:
                boxes_x.append((start, end))
            in_run = False
    if in_run:
        end = w-1
        if end - start >= 5:
            boxes_x.append((start, end))

    out = []
    for x1, x2 in boxes_x:
        region = bin_img[:, x1:x2+1]
        rows = (region == 0).any(axis=1)
        ys = np.where(rows)[0]
        if ys.size == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max())
        out.append((int(x1), int(y1), int(x2), int(y2)))
    return out

def classify_digit(bin_digit: np.ndarray):
    h, w = bin_digit.shape
    if h < 10 or w < 6:
        return None, None

    def roi(xa, ya, xb, yb):
        x1 = int(_clamp(round(xa*w), 0, w-1))
        x2 = int(_clamp(round(xb*w), 0, w-1))
        y1 = int(_clamp(round(ya*h), 0, h-1))
        y2 = int(_clamp(round(yb*h), 0, h-1))
        if x2 <= x1: x2 = min(w-1, x1+1)
        if y2 <= y1: y2 = min(h-1, y1+1)
        return bin_digit[y1:y2, x1:x2]

    seg_rois = {
        "a": roi(0.20, 0.05, 0.80, 0.18),
        "b": roi(0.70, 0.15, 0.95, 0.50),
        "c": roi(0.70, 0.50, 0.95, 0.85),
        "d": roi(0.20, 0.82, 0.80, 0.95),
        "e": roi(0.05, 0.50, 0.30, 0.85),
        "f": roi(0.05, 0.15, 0.30, 0.50),
        "g": roi(0.20, 0.43, 0.80, 0.57),
    }

    def on(seg):
        r = seg_rois[seg]
        ratio = float(np.mean(r == 0))
        return (1 if ratio > 0.18 else 0), ratio

    bits = []
    ratios = {}
    for seg in ["a","b","c","d","e","f","g"]:
        b, r = on(seg)
        bits.append(b)
        ratios[seg] = r

    key = tuple(bits)
    return SEGMENT_MAP.get(key), {"bits": key, "ratios": ratios}

def ocr_sevenseg(bin_img: np.ndarray, expected_digits: Optional[int]):
    boxes = find_digit_boxes(bin_img)
    digits = []
    seginfo = []
    used_boxes = []
    for (x1,y1,x2,y2) in boxes:
        dimg = bin_img[y1:y2+1, x1:x2+1]
        d, info = classify_digit(dimg)
        if d is None:
            continue
        digits.append(d)
        seginfo.append(info)
        used_boxes.append((x1,y1,x2,y2))

    value = "".join(digits) if digits else ""
    ok = True
    if expected_digits is not None and expected_digits > 0 and len(digits) != expected_digits:
        ok = False
    return {"value": value, "digits": digits, "boxes": used_boxes, "segments": seginfo, "ok": ok, "found": len(digits)}

class SevenSegCoordinator(DataUpdateCoordinator):
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        self.entry = entry
        self.camera_entity = entry.data[CONF_CAMERA]
        interval = int(entry.data.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL))
        super().__init__(hass, _LOGGER, name=f"{DOMAIN}:{entry.entry_id}", update_interval=timedelta(seconds=interval))
        # Last-known-value cache
        self._last_value = None
        self._last_success = None
        self._last_error = None


    async def _async_update_data(self):

        try:

                    # Cache last known good result so the sensor keeps a value even if the camera stream flakes out.

                    cache = self.hass.data.setdefault(DOMAIN, {}).setdefault(self.entry.entry_id, {})

                    last_good = cache.get("last")


                    try:

                        image = await async_get_image(self.hass, self.camera_entity)

                        if image is None:

                            raise UpdateFailed("Unable to get image (None)")


                        im = Image.open(BytesIO(image.content)).convert("RGB")


                        # Crop

                        w_img, h_img = im.size

                        x = int(self.entry.data.get(CONF_CROP_X, 0))

                        y = int(self.entry.data.get(CONF_CROP_Y, 0))

                        w = int(self.entry.data.get(CONF_CROP_W, 0))

                        h = int(self.entry.data.get(CONF_CROP_H, 0))

                        x = _clamp(x, 0, w_img-1)

                        y = _clamp(y, 0, h_img-1)

                        w = _clamp(w if w > 0 else w_img-x, 1, w_img-x)

                        h = _clamp(h if h > 0 else h_img-y, 1, h_img-y)


                        im = im.crop((x, y, x+w, y+h))

                        rotate = int(self.entry.data.get(CONF_ROTATE, 0))

                        im = _rotate_pil(im, rotate)


                        gray = im.convert("L")

                        if bool(self.entry.data.get(CONF_AUTOCONTRAST, True)):

                            gray = ImageOps.autocontrast(gray)


                        blur_sigma = float(self.entry.data.get(CONF_BLUR, 1.2))

                        if blur_sigma > 0:

                            gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_sigma))


                        arr = np.array(gray, dtype=np.uint8)

                        block = int(self.entry.data.get(CONF_BLOCK_SIZE, 41))

                        c_val = int(self.entry.data.get(CONF_C, 5))


                        bin_img = adaptive_threshold_mean(arr, block=block, c=c_val)

                        bin_img = normalize_polarity(bin_img, force_invert=bool(self.entry.data.get(CONF_FORCE_INVERT, False)))

                        bin_img = clear_border(bin_img, px=int(self.entry.data.get(CONF_BORDER_CLEAR, 10)))

                        bin_img = despeckle(bin_img, min_area=int(self.entry.data.get(CONF_MIN_AREA, 30)))


                        expected = int(self.entry.data.get(CONF_EXPECTED_DIGITS, 0)) or None

                        res = ocr_sevenseg(bin_img, expected_digits=expected)
            # Output type + monotonic guard
            output_type = str(self.entry.data.get(CONF_OUTPUT_TYPE, DEFAULT_OUTPUT_TYPE))
            allow_decrease = bool(self.entry.data.get(CONF_ALLOW_DECREASE, DEFAULT_ALLOW_DECREASE))
            parsed, raw = _parse_value(str(res.get('value','')), output_type)
            res['raw_ocr'] = raw
            res['output_type'] = output_type
            # If numeric, enforce monotonic unless explicitly allowed
            if output_type in ('int','float') and (self._last_value is not None) and (not allow_decrease):
                try:
                    last_num = float(self._last_value)
                    new_num = float(parsed)
                    if new_num < last_num:
                        return {
                            'value': self._last_value,
                            'last_update_success': False,
                            'last_error': 'decrease_detected',
                            'rejected_reason': 'decrease_detected',
                            'last_success': self._last_success,
                            'raw_ocr': raw,
                            'output_type': output_type,
                            'allow_decrease': allow_decrease,
                        }
                except Exception:
                    pass
            res['value'] = parsed
            res['allow_decrease'] = allow_decrease

                        res["crop"] = {"x": x, "y": y, "w": w, "h": h, "rotate": rotate}

                        res["preprocess"] = {

                            "autocontrast": bool(self.entry.data.get(CONF_AUTOCONTRAST, True)),

                            "blur_sigma": blur_sigma,

                            "block_size": block,

                            "c": c_val,

                            "border_clear": int(self.entry.data.get(CONF_BORDER_CLEAR, 10)),

                            "min_area": int(self.entry.data.get(CONF_MIN_AREA, 30)),

                            "force_invert": bool(self.entry.data.get(CONF_FORCE_INVERT, False)),

                        }

                        res["image_error"] = None


                        # cache good result

                        cache["last"] = res

                        return res


                    except Exception as err:

                        _LOGGER.error("Error fetching %s data: %s", self.name, err)


                        if last_good:

                            # Return cached value but mark error; this keeps the entity visible and the last value usable.

                            cached = dict(last_good)

                            cached["image_error"] = str(err)

                            cached["ok"] = False

                            return cached


                        raise UpdateFailed(str(err)) from err


            class SevenSegSensor(Entity):

                _attr_has_entity_name = True

                _attr_name = DEFAULT_NAME

                _attr_icon = "mdi:counter"

        except Exception as err:

            self._last_error = str(err)

            if self._last_value is not None:

                return {

                    'value': self._last_value,

                    'last_update_success': False,

                    'last_error': self._last_error,

                    'last_success': self._last_success,

                }

            raise


    def __init__(self, coordinator: SevenSegCoordinator, entry: ConfigEntry):
        self.coordinator = coordinator
        self.entry = entry
        self._attr_unique_id = f"{DOMAIN}:{entry.entry_id}"

    @property
    def available(self) -> bool:
        # Keep entity available if we have a cached value.
        data = self.coordinator.data or {}
        return self.coordinator.last_update_success or bool(data.get('value'))

    @property
    def state(self):
        data = self.coordinator.data or {}
        return data.get("value", "")

    @property
    def extra_state_attributes(self):
        data = self.coordinator.data or {}
        return {
            "digits": data.get("digits"),
            "found": data.get("found"),
            "ok": data.get("ok"),
            "boxes": data.get("boxes"),
            "segments": data.get("segments"),
            "crop": data.get("crop"),
            "preprocess": data.get("preprocess"),
            "camera": self.coordinator.camera_entity,
        }

    async def async_added_to_hass(self) -> None:
        self.async_on_remove(self.coordinator.async_add_listener(self.async_write_ha_state))

    async def async_update(self) -> None:
        await self.coordinator.async_request_refresh()

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback):
    coordinator = SevenSegCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    async_add_entities([SevenSegSensor(coordinator, entry)], True)

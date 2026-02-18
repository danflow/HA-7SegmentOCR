from __future__ import annotations

from datetime import timedelta
import logging
from dataclasses import dataclass

import numpy as np
import cv2

from homeassistant.components.camera import async_get_image
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.helpers.entity import Entity

from .const import (
    DOMAIN, DEFAULT_NAME,
    CONF_CAMERA, CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL,
    CONF_CROP_X, CONF_CROP_Y, CONF_CROP_W, CONF_CROP_H, CONF_ROTATE,
    CONF_EXPECTED_DIGITS,
    CONF_CLAHE_CLIP, CONF_CLAHE_TILE, CONF_BLUR,
    CONF_ADAPT_METHOD, CONF_BLOCK_SIZE, CONF_C,
    CONF_BORDER_CLEAR, CONF_MIN_AREA, CONF_FORCE_INVERT,
)

_LOGGER = logging.getLogger(__name__)

SEGMENT_MAP = {
    # a,b,c,d,e,f,g (top, ur, lr, bottom, ll, ul, mid) -> digit
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

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def despeckle(binary: np.ndarray, min_area: int, foreground_val: int = 0) -> np.ndarray:
    if min_area <= 0:
        return binary
    mask = (binary == foreground_val).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = binary.copy()
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            out[labels == i] = 255 - foreground_val
    return out

def preprocess(
    bgr: np.ndarray,
    crop: tuple[int,int,int,int],
    rotate: int,
    clahe_clip: float,
    clahe_tile: int,
    blur: int,
    adapt_method: str,
    block_size: int,
    c_value: int,
    border_clear: int,
    min_area: int,
    force_invert: bool,
):
    h_img, w_img = bgr.shape[:2]
    x,y,w,h = crop
    x = _clamp(int(x), 0, w_img-1)
    y = _clamp(int(y), 0, h_img-1)
    w = _clamp(int(w) if w>0 else w_img-x, 1, w_img-x)
    h = _clamp(int(h) if h>0 else h_img-y, 1, h_img-y)

    roi = bgr[y:y+h, x:x+w].copy()

    r = int(rotate) % 360
    if r == 90:
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    elif r == 180:
        roi = cv2.rotate(roi, cv2.ROTATE_180)
    elif r == 270:
        roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    clahe_tile = _clamp(int(clahe_tile), 2, 64)
    clahe_clip = float(max(0.1, min(10.0, float(clahe_clip))))
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    gray2 = clahe.apply(gray)

    blur = _clamp(int(blur), 0, 31)
    if blur != 0 and blur % 2 == 0:
        blur += 1
    if blur >= 3:
        gray2 = cv2.GaussianBlur(gray2, (blur, blur), 0)

    block_size = _clamp(int(block_size), 3, 99)
    if block_size % 2 == 0:
        block_size += 1
    c_value = _clamp(int(c_value), -50, 50)

    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if adapt_method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    bin_img = cv2.adaptiveThreshold(gray2, 255, method, cv2.THRESH_BINARY, block_size, c_value)

    # normalize to white background
    if float(bin_img.mean()) < 127:
        bin_img = cv2.bitwise_not(bin_img)

    if force_invert:
        bin_img = cv2.bitwise_not(bin_img)

    border_clear = _clamp(int(border_clear), 0, 30)
    if border_clear > 0:
        bin_img[:border_clear, :] = 255
        bin_img[-border_clear:, :] = 255
        bin_img[:, :border_clear] = 255
        bin_img[:, -border_clear:] = 255

    min_area = _clamp(int(min_area), 0, 5000)
    bin_img = despeckle(bin_img, min_area=min_area, foreground_val=0)

    return bin_img, {"x":x,"y":y,"w":w,"h":h,"rotate":r}

def find_digit_boxes(bin_img: np.ndarray):
    # bin_img: 0 digits, 255 bg
    h, w = bin_img.shape[:2]
    col_has_fg = (bin_img == 0).any(axis=0).astype(np.uint8)
    boxes = []
    in_run = False
    start = 0
    for x in range(w):
        if col_has_fg[x] and not in_run:
            in_run = True
            start = x
        elif not col_has_fg[x] and in_run:
            end = x-1
            if end - start >= 5:
                boxes.append((start, end))
            in_run = False
    if in_run:
        end = w-1
        if end - start >= 5:
            boxes.append((start, end))

    # For each x-run, find y bounds
    out = []
    for (x1, x2) in boxes:
        region = bin_img[:, x1:x2+1]
        rows = (region == 0).any(axis=1)
        ys = np.where(rows)[0]
        if ys.size == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max())
        out.append((int(x1), int(y1), int(x2), int(y2)))
    return out

def classify_digit(bin_digit: np.ndarray):
    # expects white bg 255 and black segments 0
    h, w = bin_digit.shape[:2]
    if h < 10 or w < 6:
        return None, None

    # Define segment ROIs (fractions tuned for typical 7-seg)
    # a(top), b(ur), c(lr), d(bottom), e(ll), f(ul), g(mid)
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
        # percent of black pixels
        ratio = float(np.mean(r == 0))
        return 1 if ratio > 0.18 else 0, ratio

    bits = []
    ratios = {}
    for seg in ["a","b","c","d","e","f","g"]:
        b, r = on(seg)
        bits.append(b)
        ratios[seg] = r
    key = tuple(bits)
    return SEGMENT_MAP.get(key), {"bits": key, "ratios": ratios}

def ocr_sevenseg(bin_img: np.ndarray, expected_digits: int | None = None):
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
    return {
        "value": value,
        "digits": digits,
        "boxes": used_boxes,
        "segments": seginfo,
        "ok": ok,
        "found": len(digits),
    }


class SevenSegCoordinator(DataUpdateCoordinator):
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        self.entry = entry
        self.camera_entity = entry.data[CONF_CAMERA]
        interval = int(entry.data.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL))
        super().__init__(
            hass,
            _LOGGER,
            name=f"{DOMAIN}:{entry.entry_id}",
            update_interval=timedelta(seconds=interval),
        )

    async def _async_update_data(self):
        try:
            image = await async_get_image(self.hass, self.camera_entity)
            if image is None:
                raise UpdateFailed("Keine Kamera-Bilddaten erhalten")
            content = image.content
            arr = np.frombuffer(content, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise UpdateFailed("OpenCV konnte das Bild nicht dekodieren")

            crop = (
                int(self.entry.data.get(CONF_CROP_X, 0)),
                int(self.entry.data.get(CONF_CROP_Y, 0)),
                int(self.entry.data.get(CONF_CROP_W, 0)),
                int(self.entry.data.get(CONF_CROP_H, 0)),
            )
            rotate = int(self.entry.data.get(CONF_ROTATE, 0))
            bin_img, crop_meta = preprocess(
                bgr=bgr,
                crop=crop,
                rotate=rotate,
                clahe_clip=float(self.entry.data.get(CONF_CLAHE_CLIP, 2.0)),
                clahe_tile=int(self.entry.data.get(CONF_CLAHE_TILE, 8)),
                blur=int(self.entry.data.get(CONF_BLUR, 5)),
                adapt_method=str(self.entry.data.get(CONF_ADAPT_METHOD, "gaussian")),
                block_size=int(self.entry.data.get(CONF_BLOCK_SIZE, 41)),
                c_value=int(self.entry.data.get(CONF_C, 5)),
                border_clear=int(self.entry.data.get(CONF_BORDER_CLEAR, 10)),
                min_area=int(self.entry.data.get(CONF_MIN_AREA, 30)),
                force_invert=bool(self.entry.data.get(CONF_FORCE_INVERT, False)),
            )

            expected = int(self.entry.data.get(CONF_EXPECTED_DIGITS, 0)) or None
            res = ocr_sevenseg(bin_img, expected_digits=expected)

            res["crop"] = crop_meta
            res["preprocess"] = {
                "clahe_clip": float(self.entry.data.get(CONF_CLAHE_CLIP, 2.0)),
                "clahe_tile": int(self.entry.data.get(CONF_CLAHE_TILE, 8)),
                "blur": int(self.entry.data.get(CONF_BLUR, 5)),
                "adapt_method": str(self.entry.data.get(CONF_ADAPT_METHOD, "gaussian")),
                "block_size": int(self.entry.data.get(CONF_BLOCK_SIZE, 41)),
                "c": int(self.entry.data.get(CONF_C, 5)),
                "border_clear": int(self.entry.data.get(CONF_BORDER_CLEAR, 10)),
                "min_area": int(self.entry.data.get(CONF_MIN_AREA, 30)),
                "force_invert": bool(self.entry.data.get(CONF_FORCE_INVERT, False)),
            }
            return res
        except Exception as err:
            raise UpdateFailed(str(err)) from err


class SevenSegSensor(Entity):
    _attr_has_entity_name = True
    _attr_name = DEFAULT_NAME
    _attr_icon = "mdi:counter"

    def __init__(self, coordinator: SevenSegCoordinator, entry: ConfigEntry):
        self.coordinator = coordinator
        self.entry = entry
        self._attr_unique_id = f"{DOMAIN}:{entry.entry_id}"

    @property
    def available(self) -> bool:
        return self.coordinator.last_update_success

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

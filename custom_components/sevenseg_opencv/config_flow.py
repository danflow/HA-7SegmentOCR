from __future__ import annotations

import json
import voluptuous as vol

from homeassistant import config_entries

from .const import (
    DOMAIN,
    DEFAULT_NAME,
    DEFAULT_SCAN_INTERVAL,
    DEFAULT_OUTPUT_TYPE, DEFAULT_ALLOW_DECREASE,
    CONF_CAMERA,
    CONF_CROP_X, CONF_CROP_Y, CONF_CROP_W, CONF_CROP_H,
    CONF_ROTATE,
    CONF_EXPECTED_DIGITS,
    CONF_SCAN_INTERVAL,
    CONF_AUTOCONTRAST, CONF_BLUR, CONF_BLOCK_SIZE, CONF_C,
    CONF_BORDER_CLEAR, CONF_MIN_AREA, CONF_FORCE_INVERT,
    CONF_ALLOW_DECREASE, CONF_OUTPUT_TYPE,
)

CONF_PRESET_JSON = "preset_json"
ROTATE_OPTIONS = [0, 90, 180, 270]


def _merge_preset(data: dict, preset_json: str) -> dict:
    preset_json = (preset_json or "").strip()
    if not preset_json:
        return data
    try:
        preset = json.loads(preset_json)
        if not isinstance(preset, dict):
            return data
    except Exception:
        return data

    allowed = {
        CONF_CAMERA, CONF_SCAN_INTERVAL, CONF_EXPECTED_DIGITS,
        CONF_CROP_X, CONF_CROP_Y, CONF_CROP_W, CONF_CROP_H, CONF_ROTATE,
        CONF_AUTOCONTRAST, CONF_BLUR, CONF_BLOCK_SIZE, CONF_C, CONF_BORDER_CLEAR, CONF_MIN_AREA, CONF_FORCE_INVERT,
    CONF_ALLOW_DECREASE, CONF_OUTPUT_TYPE,
    }
    for k, v in preset.items():
        if k in allowed:
            data[k] = v
    return data


class SevenSegPureConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        if user_input is not None:
            data = dict(user_input)
            data = _merge_preset(data, data.get(CONF_PRESET_JSON, ""))

            # Normalize rotate
            try:
                data[CONF_ROTATE] = int(data.get(CONF_ROTATE, 0))
            except Exception:
                data[CONF_ROTATE] = 0

            await self.async_set_unique_id(f"{DOMAIN}:{data[CONF_CAMERA]}")
            self._abort_if_unique_id_configured()
            data.pop(CONF_PRESET_JSON, None)
            return self.async_create_entry(title=DEFAULT_NAME, data=data)

        schema = vol.Schema({
            vol.Required(CONF_CAMERA): str,  # entity_id, e.g. camera.watermeter
            vol.Optional(CONF_SCAN_INTERVAL, default=DEFAULT_SCAN_INTERVAL): vol.Coerce(int),
            vol.Optional(CONF_EXPECTED_DIGITS, default=5): vol.Coerce(int),

            vol.Optional(CONF_PRESET_JSON, default=""): str,

            vol.Optional(CONF_CROP_X, default=0): vol.Coerce(int),
            vol.Optional(CONF_CROP_Y, default=0): vol.Coerce(int),
            vol.Optional(CONF_CROP_W, default=0): vol.Coerce(int),
            vol.Optional(CONF_CROP_H, default=0): vol.Coerce(int),
            vol.Optional(CONF_ROTATE, default=0): vol.In(ROTATE_OPTIONS),

            vol.Optional(CONF_AUTOCONTRAST, default=True): bool,
            vol.Optional(CONF_BLUR, default=1.2): vol.Coerce(float),
            vol.Optional(CONF_BLOCK_SIZE, default=41): vol.Coerce(int),
            vol.Optional(CONF_C, default=5): vol.Coerce(int),
            vol.Optional(CONF_BORDER_CLEAR, default=10): vol.Coerce(int),
            vol.Optional(CONF_MIN_AREA, default=30): vol.Coerce(int),
            vol.Optional(CONF_FORCE_INVERT, default=False): bool,

            vol.Optional(CONF_OUTPUT_TYPE, default=DEFAULT_OUTPUT_TYPE): vol.In(["string","int","float"]),
            vol.Optional(CONF_ALLOW_DECREASE, default=DEFAULT_ALLOW_DECREASE): bool,
        })
        return self.async_show_form(step_id="user", data_schema=schema)

    @staticmethod
    def async_get_options_flow(config_entry):
        return SevenSegPureOptionsFlow(config_entry)


class SevenSegPureOptionsFlow(config_entries.OptionsFlow):
    def __init__(self, config_entry):
        self._config_entry = config_entry
        self._data = dict(config_entry.data)

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            data = dict(self._data)
            data.update(user_input)
            data = _merge_preset(data, data.get(CONF_PRESET_JSON, ""))

            try:
                data[CONF_ROTATE] = int(data.get(CONF_ROTATE, 0))
            except Exception:
                data[CONF_ROTATE] = 0

            data.pop(CONF_PRESET_JSON, None)
            return self.async_create_entry(title="", data=data)

        d = self._data
        schema = vol.Schema({
            vol.Optional(CONF_SCAN_INTERVAL, default=d.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)): vol.Coerce(int),
            vol.Optional(CONF_EXPECTED_DIGITS, default=d.get(CONF_EXPECTED_DIGITS, 5)): vol.Coerce(int),

            vol.Optional(CONF_PRESET_JSON, default=""): str,

            vol.Optional(CONF_CROP_X, default=d.get(CONF_CROP_X, 0)): vol.Coerce(int),
            vol.Optional(CONF_CROP_Y, default=d.get(CONF_CROP_Y, 0)): vol.Coerce(int),
            vol.Optional(CONF_CROP_W, default=d.get(CONF_CROP_W, 0)): vol.Coerce(int),
            vol.Optional(CONF_CROP_H, default=d.get(CONF_CROP_H, 0)): vol.Coerce(int),
            vol.Optional(CONF_ROTATE, default=int(d.get(CONF_ROTATE, 0))): vol.In(ROTATE_OPTIONS),

            vol.Optional(CONF_AUTOCONTRAST, default=bool(d.get(CONF_AUTOCONTRAST, True))): bool,
            vol.Optional(CONF_BLUR, default=float(d.get(CONF_BLUR, 1.2))): vol.Coerce(float),
            vol.Optional(CONF_BLOCK_SIZE, default=int(d.get(CONF_BLOCK_SIZE, 41))): vol.Coerce(int),
            vol.Optional(CONF_C, default=int(d.get(CONF_C, 5))): vol.Coerce(int),
            vol.Optional(CONF_BORDER_CLEAR, default=int(d.get(CONF_BORDER_CLEAR, 10))): vol.Coerce(int),
            vol.Optional(CONF_MIN_AREA, default=int(d.get(CONF_MIN_AREA, 30))): vol.Coerce(int),
            vol.Optional(CONF_FORCE_INVERT, default=bool(d.get(CONF_FORCE_INVERT, False))): bool,

            vol.Optional(CONF_OUTPUT_TYPE, default=d.get(CONF_OUTPUT_TYPE, DEFAULT_OUTPUT_TYPE)): vol.In(["string","int","float"]),
            vol.Optional(CONF_ALLOW_DECREASE, default=bool(d.get(CONF_ALLOW_DECREASE, DEFAULT_ALLOW_DECREASE))): bool,
        })
        return self.async_show_form(step_id="init", data_schema=schema)

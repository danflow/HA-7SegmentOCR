from __future__ import annotations

import json
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.helpers import selector

from .const import (
    DOMAIN,
    DEFAULT_NAME,
    DEFAULT_SCAN_INTERVAL,
    CONF_CAMERA,
    CONF_CROP_X, CONF_CROP_Y, CONF_CROP_W, CONF_CROP_H,
    CONF_ROTATE,
    CONF_EXPECTED_DIGITS,
    CONF_SCAN_INTERVAL,
    CONF_AUTOCONTRAST, CONF_BLUR, CONF_BLOCK_SIZE, CONF_C,
    CONF_BORDER_CLEAR, CONF_MIN_AREA, CONF_FORCE_INVERT,
)

ROTATE_OPTIONS = [0, 90, 180, 270]

CONF_PRESET_JSON = "preset_json"

def _num(min_v, max_v, step=1, unit=None, mode="box"):
    return selector.NumberSelector(
        selector.NumberSelectorConfig(min=min_v, max=max_v, step=step, unit_of_measurement=unit, mode=mode)
    )

def _bool():
    return selector.BooleanSelector()

def _text(multiline=False):
    return selector.TextSelector(selector.TextSelectorConfig(multiline=multiline))

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

    # Accept keys from our consts only
    allowed = {
        CONF_CAMERA, CONF_SCAN_INTERVAL, CONF_EXPECTED_DIGITS,
        CONF_CROP_X, CONF_CROP_Y, CONF_CROP_W, CONF_CROP_H, CONF_ROTATE,
        CONF_AUTOCONTRAST, CONF_BLUR, CONF_BLOCK_SIZE, CONF_C, CONF_BORDER_CLEAR, CONF_MIN_AREA, CONF_FORCE_INVERT,
    }
    for k, v in preset.items():
        if k in allowed:
            data[k] = v
    return data

class SevenSegPureConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    def __init__(self):
        self._data: dict = {}

    async def async_step_user(self, user_input=None):
        if user_input is not None:
            self._data.update(user_input)
            # Optional preset JSON can already fill crop/preprocess.
            self._data = _merge_preset(self._data, self._data.get(CONF_PRESET_JSON, ""))
            return await self.async_step_crop()

        schema = vol.Schema({
            vol.Required(CONF_CAMERA): selector.EntitySelector(selector.EntitySelectorConfig(domain="camera")),
            vol.Optional(CONF_SCAN_INTERVAL, default=DEFAULT_SCAN_INTERVAL): _num(5, 3600, step=1, unit="s"),
            vol.Optional(CONF_EXPECTED_DIGITS, default=5): _num(1, 12, step=1),
            vol.Optional(CONF_PRESET_JSON): _text(multiline=True),
        })
        return self.async_show_form(step_id="user", data_schema=schema, description_placeholders={})

    async def async_step_crop(self, user_input=None):
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_preprocess()

        schema = vol.Schema({
            vol.Optional(CONF_CROP_X, default=0): _num(0, 5000, step=1),
            vol.Optional(CONF_CROP_Y, default=0): _num(0, 5000, step=1),
            vol.Optional(CONF_CROP_W, default=0): _num(0, 5000, step=1),
            vol.Optional(CONF_CROP_H, default=0): _num(0, 5000, step=1),
            vol.Optional(CONF_ROTATE, default="0"): selector.SelectSelector(
                selector.SelectSelectorConfig(options=[str(x) for x in ROTATE_OPTIONS], mode=selector.SelectSelectorMode.DROPDOWN)
            ),
        })
        return self.async_show_form(
            step_id="crop",
            data_schema=schema,
            description_placeholders={})

    async def async_step_preprocess(self, user_input=None):
        if user_input is not None:
            self._data.update(user_input)
            return await self.async_step_import()

        schema = vol.Schema({
            vol.Optional(CONF_AUTOCONTRAST, default=True): _bool(),
            vol.Optional(CONF_BLUR, default=1.2): _num(0.0, 5.0, step=0.1, mode="slider"),
            vol.Optional(CONF_BLOCK_SIZE, default=41): _num(3, 199, step=2),
            vol.Optional(CONF_C, default=5): _num(-50, 50, step=1),
            vol.Optional(CONF_BORDER_CLEAR, default=10): _num(0, 50, step=1),
            vol.Optional(CONF_MIN_AREA, default=30): _num(0, 5000, step=1),
            vol.Optional(CONF_FORCE_INVERT, default=False): _bool(),
        })
        return self.async_show_form(
            step_id="preprocess",
            data_schema=schema)

    async def async_step_import(self, user_input=None):
        if user_input is not None:
            preset_json = user_input.get(CONF_PRESET_JSON, "")
            data = dict(self._data)

            # convert rotate select back to int if needed
            if isinstance(data.get(CONF_ROTATE), str) and str(data[CONF_ROTATE]).isdigit():
                data[CONF_ROTATE] = int(data[CONF_ROTATE])

            data = _merge_preset(data, preset_json)

            await self.async_set_unique_id(f"{DOMAIN}:{data[CONF_CAMERA]}")
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title=DEFAULT_NAME, data=data)

        schema = vol.Schema({
            vol.Optional(CONF_PRESET_JSON): _text(multiline=True),
        })
        return self.async_show_form(
            step_id="import",
            data_schema=schema)

    @staticmethod
    def async_get_options_flow(config_entry):
        return SevenSegPureOptionsFlow(config_entry)

class SevenSegPureOptionsFlow(config_entries.OptionsFlow):
    def __init__(self, config_entry):
        self.config_entry = config_entry
        self._data = dict(config_entry.data)

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            preset_json = user_input.pop(CONF_PRESET_JSON, "")
            self._data.update(user_input)
            # Rotate comes from selector as string
            if isinstance(self._data.get(CONF_ROTATE), str) and str(self._data[CONF_ROTATE]).isdigit():
                self._data[CONF_ROTATE] = int(self._data[CONF_ROTATE])
            self._data = _merge_preset(self._data, preset_json)
            return self.async_create_entry(title="", data=self._data)

        d = self._data
        schema = vol.Schema({
            vol.Optional(CONF_SCAN_INTERVAL, default=d.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)): _num(5, 3600, step=1, unit="s"),
            vol.Optional(CONF_EXPECTED_DIGITS, default=d.get(CONF_EXPECTED_DIGITS, 5)): _num(1, 12, step=1),

            vol.Optional(CONF_CROP_X, default=d.get(CONF_CROP_X, 0)): _num(0, 5000, step=1),
            vol.Optional(CONF_CROP_Y, default=d.get(CONF_CROP_Y, 0)): _num(0, 5000, step=1),
            vol.Optional(CONF_CROP_W, default=d.get(CONF_CROP_W, 0)): _num(0, 5000, step=1),
            vol.Optional(CONF_CROP_H, default=d.get(CONF_CROP_H, 0)): _num(0, 5000, step=1),
            vol.Optional(CONF_ROTATE, default=str(int(d.get(CONF_ROTATE, 0)))): selector.SelectSelector(
                selector.SelectSelectorConfig(options=[str(x) for x in ROTATE_OPTIONS], mode=selector.SelectSelectorMode.DROPDOWN)
            ),

            vol.Optional(CONF_AUTOCONTRAST, default=bool(d.get(CONF_AUTOCONTRAST, True))): _bool(),
            vol.Optional(CONF_BLUR, default=float(d.get(CONF_BLUR, 1.2))): _num(0.0, 5.0, step=0.1, mode="slider"),
            vol.Optional(CONF_BLOCK_SIZE, default=int(d.get(CONF_BLOCK_SIZE, 41))): _num(3, 199, step=2),
            vol.Optional(CONF_C, default=int(d.get(CONF_C, 5))): _num(-50, 50, step=1),
            vol.Optional(CONF_BORDER_CLEAR, default=int(d.get(CONF_BORDER_CLEAR, 10))): _num(0, 50, step=1),
            vol.Optional(CONF_MIN_AREA, default=int(d.get(CONF_MIN_AREA, 30))): _num(0, 5000, step=1),
            vol.Optional(CONF_FORCE_INVERT, default=bool(d.get(CONF_FORCE_INVERT, False))): _bool(),

            vol.Optional(CONF_PRESET_JSON): _text(multiline=True),
        })
        return self.async_show_form(step_id="init", data_schema=schema)

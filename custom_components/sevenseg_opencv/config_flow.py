from __future__ import annotations

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

class SevenSegPureConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input=None):
        if user_input is not None:
            await self.async_set_unique_id(f"{DOMAIN}:{user_input[CONF_CAMERA]}")
            self._abort_if_unique_id_configured()
            return self.async_create_entry(title=DEFAULT_NAME, data=user_input)

        schema = vol.Schema({
            vol.Required(CONF_CAMERA): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="camera")
            ),
            vol.Optional(CONF_SCAN_INTERVAL, default=DEFAULT_SCAN_INTERVAL): vol.Coerce(int),
            vol.Optional(CONF_EXPECTED_DIGITS, default=5): vol.Coerce(int),

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
        })
        return self.async_show_form(step_id="user", data_schema=schema)

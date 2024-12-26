# The proxy_thermostat component.  # noqa: D104

from __future__ import annotations

import logging

from homeassistant.const import Platform

DOMAIN = "proxy_thermostat"
PLATFORMS = [Platform.CLIMATE]

_LOGGER = logging.getLogger(__name__)
_LOGGER.info("COM-16: Starting")

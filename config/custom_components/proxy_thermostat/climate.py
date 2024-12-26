"""Adds support for Proxy thermostat units."""

from __future__ import annotations

import asyncio
import datetime
import logging
import math
from typing import Any

import voluptuous as vol

from homeassistant.components.climate import (
    ATTR_PRESET_MODE,
    PLATFORM_SCHEMA,
    PRESET_ACTIVITY,
    PRESET_AWAY,
    PRESET_COMFORT,
    PRESET_HOME,
    PRESET_NONE,
    PRESET_SLEEP,
    ClimateEntity,
    ClimateEntityFeature,
    HVACAction,
    HVACMode,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    ATTR_TEMPERATURE,
    CONF_NAME,
    CONF_UNIQUE_ID,
    EVENT_HOMEASSISTANT_START,
    PRECISION_HALVES,
    PRECISION_TENTHS,
    PRECISION_WHOLE,
    STATE_ON,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import CoreState, HomeAssistant, State, callback
from homeassistant.exceptions import ConditionError
from homeassistant.helpers import condition
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import (
    EventStateChangedData,
    async_track_state_change_event,
    async_track_time_interval,
)
from homeassistant.helpers.reload import async_setup_reload_service
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType, EventType

from . import DOMAIN, PLATFORMS

_LOGGER = logging.getLogger(__name__)

DEFAULT_TOLERANCE = 0.3
DEFAULT_NAME = "Proxy Thermostat"

CONF_HEATER = "heater"
CONF_SENSOR = "target_sensor"
CONF_MIN_TEMP = "min_temp"
CONF_MAX_TEMP = "max_temp"
CONF_TARGET_TEMP = "target_temp"
CONF_AC_MODE = "ac_mode"
CONF_MIN_DUR = "min_cycle_duration"
CONF_COLD_TOLERANCE = "cold_tolerance"
CONF_HOT_TOLERANCE = "hot_tolerance"
CONF_KEEP_ALIVE = "keep_alive"
CONF_ONLY_CHANGED = "only_changed"
CONF_INITIAL_HVAC_MODE = "initial_hvac_mode"
CONF_PRECISION = "precision"
CONF_TEMP_STEP = "target_temp_step"

CONF_PRESETS = {
    p: f"{p}_temp"
    for p in (
        PRESET_AWAY,
        PRESET_COMFORT,
        PRESET_HOME,
        PRESET_SLEEP,
        PRESET_ACTIVITY,
    )
}

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_HEATER): cv.entity_id,
        vol.Required(CONF_SENSOR): cv.entity_id,
        vol.Optional(CONF_AC_MODE): cv.boolean,
        vol.Optional(CONF_MAX_TEMP): vol.Coerce(float),
        vol.Optional(CONF_MIN_DUR): cv.positive_time_period,
        vol.Optional(CONF_MIN_TEMP): vol.Coerce(float),
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_COLD_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
        vol.Optional(CONF_HOT_TOLERANCE, default=DEFAULT_TOLERANCE): vol.Coerce(float),
        vol.Optional(CONF_TARGET_TEMP): vol.Coerce(float),
        vol.Optional(CONF_KEEP_ALIVE): cv.positive_time_period,
        vol.Optional(CONF_INITIAL_HVAC_MODE): vol.In(
            [HVACMode.COOL, HVACMode.HEAT, HVACMode.OFF]
        ),
        vol.Optional(CONF_PRECISION): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        ),
        vol.Optional(CONF_TEMP_STEP): vol.In(
            [PRECISION_TENTHS, PRECISION_HALVES, PRECISION_WHOLE]
        ),
        vol.Optional(CONF_UNIQUE_ID): cv.string,
        vol.Optional(CONF_ONLY_CHANGED, default=False): cv.boolean,
    }
).extend({vol.Optional(v): vol.Coerce(float) for (k, v) in CONF_PRESETS.items()})


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the proxy thermostat platform."""

    await async_setup_reload_service(hass, DOMAIN, PLATFORMS)

    name = config.get(CONF_NAME)
    heater_entity_id = config.get(CONF_HEATER)
    sensor_entity_id = config.get(CONF_SENSOR)
    min_temp = config.get(CONF_MIN_TEMP)
    max_temp = config.get(CONF_MAX_TEMP)
    target_temp = config.get(CONF_TARGET_TEMP)
    ac_mode = config.get(CONF_AC_MODE)
    min_cycle_duration = config.get(CONF_MIN_DUR)
    cold_tolerance = config.get(CONF_COLD_TOLERANCE)
    hot_tolerance = config.get(CONF_HOT_TOLERANCE)
    keep_alive = config.get(CONF_KEEP_ALIVE)
    initial_hvac_mode = config.get(CONF_INITIAL_HVAC_MODE)
    presets = {
        key: config[value] for key, value in CONF_PRESETS.items() if value in config
    }
    precision = config.get(CONF_PRECISION)
    target_temperature_step = config.get(CONF_TEMP_STEP)
    unit = hass.config.units.temperature_unit
    unique_id = config.get(CONF_UNIQUE_ID)
    only_changed = config.get(CONF_ONLY_CHANGED)

    async_add_entities(
        [
            ProxyThermostat(
                name,
                heater_entity_id,
                sensor_entity_id,
                min_temp,
                max_temp,
                target_temp,
                ac_mode,
                min_cycle_duration,
                cold_tolerance,
                hot_tolerance,
                keep_alive,
                initial_hvac_mode,
                presets,
                precision,
                target_temperature_step,
                unit,
                unique_id,
                only_changed,
            )
        ]
    )


class _TempResolver:
    _max_len = 3
    _temps = []

    def add_temp(self, temp):
        if temp is None:
            return

        temp = round(temp, 1)
        if len(self._temps) > 0 and self._temps[-1] == temp:
            return
        if len(self._temps) == self._max_len:
            self._temps = self._temps[1:]
        self._temps.append(temp)

    def is_riging(self) -> bool:
        if len(self._temps) < self._max_len:
            return False
        if self._temps[2] > self._temps[0]:
            return True
        return False

    def is_descending(self) -> bool:
        if len(self._temps) < self._max_len:
            return False
        if self._temps[2] < self._temps[0]:
            return True
        return False

    def debug_inf(self) -> str:
        return f"temps:{self._temps}"

    def in_tunel(self, temp: float) -> bool:
        if len(self._temps) < self._max_len or temp is None:
            return False

        mini = min(self._temps)
        maxi = max(self._temps)
        if (maxi - mini) > 0.11:
            return False
        if temp in (mini, maxi):
            return True

        return False


class ProxyThermostat(ClimateEntity, RestoreEntity):
    """Representation of a Proxy Thermostat device."""

    _attr_should_poll = False

    def __init__(
        self,
        name,
        heater_entity_id,
        sensor_entity_id,
        min_temp,
        max_temp,
        target_temp,
        ac_mode,
        min_cycle_duration,
        cold_tolerance,
        hot_tolerance,
        keep_alive,
        initial_hvac_mode,
        presets,
        precision,
        target_temperature_step,
        unit,
        unique_id,
        only_changed,
    ):
        """Initialize the thermostat."""
        _LOGGER.debug("COM-13 [%s]: Initialize the thermostat", name)
        self._attr_name = name
        self.heater_entity_id = heater_entity_id
        self.sensor_entity_id = sensor_entity_id
        self.ac_mode = ac_mode
        self.min_cycle_duration = min_cycle_duration
        self._cold_tolerance = cold_tolerance
        self._hot_tolerance = hot_tolerance
        self._keep_alive = keep_alive
        self._hvac_mode = initial_hvac_mode
        self._saved_target_temp = target_temp or next(iter(presets.values()), None)
        self._temp_precision = precision
        self._temp_target_temperature_step = target_temperature_step
        if self.ac_mode:
            self._attr_hvac_modes = [HVACMode.COOL, HVACMode.OFF]
        else:
            self._attr_hvac_modes = [HVACMode.HEAT, HVACMode.OFF]
        self._active = False
        self._cur_temp = None
        self._temp_lock = asyncio.Lock()
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._attr_preset_mode = PRESET_NONE
        self._target_temp = target_temp
        self._attr_temperature_unit = unit
        self._attr_unique_id = unique_id
        self._attr_supported_features = ClimateEntityFeature.TARGET_TEMPERATURE
        if len(presets):
            self._attr_supported_features |= ClimateEntityFeature.PRESET_MODE
            self._attr_preset_modes = [PRESET_NONE] + list(presets.keys())
        else:
            self._attr_preset_modes = [PRESET_NONE]
        self._presets = presets
        self._only_changed = only_changed
        self._last_target_temp_set = None
        self.new_state = None
        self.old_state = None
        self.target_changed_time = None
        self._temp_resolver = _TempResolver()

    async def async_added_to_hass(self) -> None:
        """Run when entity about to be added."""
        await super().async_added_to_hass()

        # Add listener
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.sensor_entity_id], self._async_sensor_changed
            )
        )
        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self.heater_entity_id], self._async_target_climate_changed
            )
        )

        if self._keep_alive:
            self.async_on_remove(
                async_track_time_interval(
                    self.hass, self._async_control_heating, self._keep_alive
                )
            )

        @callback
        def _async_startup(*_):
            """Init on startup."""
            sensor_state = self.hass.states.get(self.sensor_entity_id)
            if sensor_state and sensor_state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            ):
                self._async_update_temp(sensor_state)
                self.async_write_ha_state()
            switch_state = self.hass.states.get(self.heater_entity_id)
            if switch_state and switch_state.state not in (
                STATE_UNAVAILABLE,
                STATE_UNKNOWN,
            ):
                self.hass.create_task(self._check_switch_initial_state())

        if self.hass.state == CoreState.running:
            _async_startup()
        else:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_START, _async_startup)

        # Check If we have an old state
        if (old_state := await self.async_get_last_state()) is not None:
            # If we have no initial temperature, restore
            if self._target_temp is None:
                # If we have a previously saved temperature
                if old_state.attributes.get(ATTR_TEMPERATURE) is None:
                    if self.ac_mode:
                        self._target_temp = self.max_temp
                    else:
                        self._target_temp = self.min_temp
                    _LOGGER.warning(
                        "Undefined target temperature, falling back to %s",
                        self._target_temp,
                    )
                else:
                    self._target_temp = float(old_state.attributes[ATTR_TEMPERATURE])
            if (
                self.preset_modes
                and old_state.attributes.get(ATTR_PRESET_MODE) in self.preset_modes
            ):
                self._attr_preset_mode = old_state.attributes.get(ATTR_PRESET_MODE)
            if not self._hvac_mode and old_state.state:
                self._hvac_mode = old_state.state

        else:
            # No previous state, try and restore defaults
            if self._target_temp is None:
                if self.ac_mode:
                    self._target_temp = self.max_temp
                else:
                    self._target_temp = self.min_temp
            _LOGGER.warning(
                "No previously saved temperature, setting to %s", self._target_temp
            )

        # Set default state to off
        if not self._hvac_mode:
            self._hvac_mode = HVACMode.OFF

    @property
    def precision(self):
        """Return the precision of the system."""
        if self._temp_precision is not None:
            return self._temp_precision
        return super().precision

    @property
    def target_temperature_step(self):
        """Return the supported step of target temperature."""
        if self._temp_target_temperature_step is not None:
            return self._temp_target_temperature_step
        # if a target_temperature_step is not defined, fallback to equal the precision
        return self.precision

    @property
    def current_temperature(self):
        """Return the sensor temperature."""
        return self._cur_temp

    @property
    def hvac_mode(self):
        """Return current operation."""
        return self._hvac_mode

    @property
    def hvac_action(self):
        """Return the current running hvac operation if supported.

        Need to be one of CURRENT_HVAC_*.
        """
        if self._hvac_mode == HVACMode.OFF:
            return HVACAction.OFF
        if not self._is_device_active:
            return HVACAction.IDLE
        if self.ac_mode:
            return HVACAction.COOLING
        return HVACAction.HEATING

    @property
    def target_temperature(self):
        """Return the temperature we try to reach."""
        return self._target_temp

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> None:
        """Set hvac mode."""
        if hvac_mode == HVACMode.HEAT:
            self._hvac_mode = HVACMode.HEAT
            await self._async_control_heating(force=True)
        elif hvac_mode == HVACMode.COOL:
            self._hvac_mode = HVACMode.COOL
            await self._async_control_heating(force=True)
        elif hvac_mode == HVACMode.OFF:
            self._hvac_mode = HVACMode.OFF
            if self._is_device_active:
                await self._async_heater_turn_off()
        else:
            _LOGGER.error("Unrecognized hvac mode: %s", hvac_mode)
            return
        # Ensure we update the current operation after changing the mode
        self.async_write_ha_state()

    async def async_set_temperature(self, **kwargs: Any) -> None:
        """Set new target temperature."""
        if (temperature := kwargs.get(ATTR_TEMPERATURE)) is None:
            return
        self._target_temp = temperature
        await self._async_control_heating(force=True)
        self.async_write_ha_state()

    @property
    def min_temp(self):
        """Return the minimum temperature."""
        if self._min_temp is not None:
            return self._min_temp

        # get default temp from super class
        return super().min_temp

    @property
    def max_temp(self):
        """Return the maximum temperature."""
        if self._max_temp is not None:
            return self._max_temp

        # Get default temp from super class
        return super().max_temp

    async def _async_sensor_changed(
        self, event: EventType[EventStateChangedData]
    ) -> None:
        """Handle temperature changes."""
        new_state = event.data["new_state"]
        if new_state is None or new_state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return

        self._async_update_temp(new_state)
        await self._async_control_heating()
        self.async_write_ha_state()

    async def _check_switch_initial_state(self):
        """Prevent the device from keep running if HVACMode.OFF."""
        if self._hvac_mode == HVACMode.OFF and self._is_device_active:
            _LOGGER.warning(
                (
                    "The climate mode is OFF, but the switch device is ON. Turning off"
                    " device %s"
                ),
                self.heater_entity_id,
            )
            await self._async_heater_turn_off()

    @callback
    def _async_target_climate_changed(
        self, event: EventType[EventStateChangedData]
    ) -> None:
        """Handle heater switch state changes."""
        self.new_state = event.data["new_state"]
        self.old_state = event.data["old_state"]
        self.target_changed_time = datetime.datetime.now()
        _LOGGER.debug(
            "COM-14 [%s]: Target therm state change: %s  NEW: %s OLD: %s",
            self._attr_name,
            self.target_changed_time,
            self.new_state,
            self.old_state,
        )
        if self.new_state is None:
            return
        if self.old_state is None:
            self.hass.create_task(self._check_switch_initial_state())
        self.async_write_ha_state()

    @callback
    def _async_update_temp(self, state: State) -> None:
        """Update thermostat with latest state from sensor."""
        try:
            cur_temp = float(state.state)
            if not math.isfinite(cur_temp):
                raise ValueError(f"Sensor has illegal state {state.state}")
            self._cur_temp = cur_temp
            self._temp_resolver.add_temp(cur_temp)
        except ValueError as ex:
            _LOGGER.error("Unable to update from sensor: %s", ex)

    async def _async_control_heating(self, time=None, force=False):
        """Check if we need to turn heating on or off."""
        async with self._temp_lock:
            if not self._active and None not in (
                self._cur_temp,
                self._target_temp,
            ):
                self._active = True
                _LOGGER.debug(
                    "COM-16 [%s]: Obtained current and target temperature. Proxy thermostat active",
                    self._attr_name,
                )
                self._print_debug_states()

            if not self._active or self._hvac_mode == HVACMode.OFF:
                return

            # If the `force` argument is True, we
            # ignore `min_cycle_duration`.
            # If the `time` argument is not none, we were invoked for
            # keep-alive purposes, and `min_cycle_duration` is irrelevant.
            if not force and time is None and self.min_cycle_duration:
                if self._is_device_active:
                    current_state = STATE_ON
                else:
                    current_state = HVACMode.OFF
                try:
                    long_enough = condition.state(
                        self.hass,
                        self.heater_entity_id,
                        current_state,
                        self.min_cycle_duration,
                    )
                except ConditionError:
                    long_enough = False

                if not long_enough:
                    return

            temp_delta = round(self._cur_temp - self._target_temp, 1)
            too_cold = self._target_temp >= self._cur_temp + self._cold_tolerance
            too_hot = self._cur_temp >= self._target_temp + self._hot_tolerance
            if self._is_device_active:
                if (self.ac_mode and too_cold) or (not self.ac_mode and too_hot):
                    if self._check_only_changed(off=True):
                        _LOGGER.info(
                            "COM-01 [%s]: Turning off heater",
                            self._attr_name,
                        )
                        self._print_debug_states()
                        await self._async_heater_turn_off(time is not None)
                    elif temp_delta > 0.5 and time is None and self._temp_is_rising():
                        _LOGGER.warning(
                            "COM-02 [%s]: force turn off due to temp delta and device still active, current=%s - target=%s > 0.5",
                            self._attr_name,
                            round(self._cur_temp, 1),
                            round(self._target_temp, 1),
                        )
                        self._print_debug_states()
                        await self._async_heater_turn_off(time is not None)
                elif self._keep_alive and self._check_only_changed(off=False):
                    if self._only_changed:
                        _LOGGER.info(
                            "COM-03 [%s]: turn on because target temp to change",
                            self._attr_name,
                        )
                    else:
                        _LOGGER.debug(
                            "COM-04 [%s]: force turn on due to keep alive",
                            self._attr_name,
                        )
                    self._print_debug_states()
                    await self._async_heater_turn_on(time is not None)
                elif not self._keep_alive and self._target_temp_not_set(off=False):
                    _LOGGER.info(
                        "COM-05 [%s]: turn on because target temp to change",
                        self._attr_name,
                    )
                    self._print_debug_states()
                    await self._async_heater_turn_on(time is not None)
                elif temp_delta < -0.5 and time is None and self._temp_is_falling():
                    _LOGGER.warning(
                        "COM-06 [%s]: force turn on due to temp delta,  target=%s - current=%s > 0.5",
                        self._attr_name,
                        round(self._target_temp, 1),
                        round(self._cur_temp, 1),
                    )
                    self._print_debug_states()
                    await self._async_heater_turn_on(time is not None)
            # device not active
            elif (self.ac_mode and too_hot) or (not self.ac_mode and too_cold):
                if self._check_only_changed(off=False):
                    _LOGGER.info("COM-15 [%s]: Turning on heater", self._attr_name)
                    await self._async_heater_turn_on(time is not None)
                elif temp_delta < -0.5 and time is None and self._temp_is_falling():
                    _LOGGER.warning(
                        "COM-07 [%s]: force turn on due to temp delta and device not active, target=%s - current=%s > 0.5",
                        self._attr_name,
                        round(self._target_temp, 1),
                        round(self._cur_temp, 1),
                    )
                    self._print_debug_states()
                    await self._async_heater_turn_on(time is not None)
            elif self._keep_alive and self._check_only_changed(off=True):
                if self._only_changed:
                    _LOGGER.info(
                        "COM-08 [%s]: turn off because target temp to chanage",
                        self._attr_name,
                    )
                else:
                    _LOGGER.debug(
                        "COM-09 [%s]: force turn off due to keep alive",
                        self._attr_name,
                    )
                self._print_debug_states()
                await self._async_heater_turn_off(time is not None)
            elif not self._keep_alive and self._target_temp_not_set(off=True):
                _LOGGER.info(
                    "COM-10 [%s]: turn off because target temp to chanage",
                    self._attr_name,
                )
                self._print_debug_states()
                await self._async_heater_turn_off(time is not None)
            elif temp_delta > 0.5 and time is None and self._temp_is_rising():
                _LOGGER.warning(
                    "COM-11 [%s]: force turn off due to temp delta, current=%s - target=%s > 0.5",
                    self._attr_name,
                    round(self._cur_temp, 1),
                    round(self._target_temp, 1),
                )
                self._print_debug_states()
                await self._async_heater_turn_off(time is not None)

    def _print_debug_states(self):
        _LOGGER.debug("COM-99 [%s]: %s", self._attr_name, self._debug_info())

    def _debug_info(self):
        return (
            f"(temp:{round(self._cur_temp,1)}, target temp:{round(self._target_temp, 1)}, target therm set to:{self._target_target_temp()}"
            f", temp resolver:{self._temp_resolver.debug_inf()})"
        )

    def _temp_is_rising(self):
        return self._temp_resolver.is_riging()

    def _temp_is_falling(self):
        return self._temp_resolver.is_descending()

    def _check_only_changed(self, off: bool) -> bool:
        return self._only_changed is False or (
            not self._temp_resolver.in_tunel(self._cur_temp)
            and self._target_temp_not_set(off)
        )

    def _target_temp_not_set(self, off: bool):
        return (off and self._target_target_temp() != self._heater_turn_off_temp()) or (
            off is False and self._target_target_temp() != self._heater_turn_on_temp()
        )

    def _target_current_temp(self):
        return (
            self.new_state.attributes["current_temperature"] if self.new_state else None
        )

    def _target_target_temp(self):
        return self.new_state.attributes["temperature"] if self.new_state else None

    @property
    def _is_device_active(self):
        """If the toggleable device is currently active."""

        state = self.hass.states.get(self.heater_entity_id)
        if not state:
            return None

        key = "hvac_action"
        if state.attributes and key in state.attributes:
            attr = state.attributes[key]
            return attr != HVACAction.IDLE
        return None

    def _heater_turn_on_temp(self):
        delta_temp = round(self.target_temperature - self.current_temperature, 1)
        if self._target_current_temp() is None:
            temp = 30
        elif delta_temp <= 0:
            temp = self._target_current_temp()
        elif delta_temp <= 0.1:
            temp = self._target_current_temp() + 0
        elif delta_temp <= 0.2:
            temp = self._target_current_temp() + 0.5
        else:
            temp = 30
        return temp

    async def _async_heater_turn_on(self, k_alive=False):
        """Turn heater  device on."""
        temp = self._heater_turn_on_temp()
        await self._async_set_target_temp(temp=temp, k_alive=k_alive)

    def _heater_turn_off_temp(self):
        delta_temp = round(self.current_temperature - self.target_temperature, 1)

        if self._target_current_temp() is None or self._hvac_mode == HVACMode.OFF:
            temp = 5
        elif delta_temp <= 0:
            temp = self._target_current_temp()
        elif delta_temp <= 0.1 or delta_temp <= 0.2:
            temp = 5
        else:
            temp = 5
        return temp

    async def _async_heater_turn_off(self, k_alive=False):
        """Turn heater  device off."""
        temp = self._heater_turn_off_temp()
        await self._async_set_target_temp(temp=temp, k_alive=k_alive)

    async def _async_set_target_temp(self, temp: float, k_alive: bool) -> None:
        self._last_target_temp_set = temp
        data = {ATTR_ENTITY_ID: self.heater_entity_id, ATTR_TEMPERATURE: temp}
        _LOGGER.debug(
            "COM-12 [%s]: Setting target thermostat tmperature: %s k_alive=%s",
            self._attr_name,
            data,
            k_alive,
        )
        await self.hass.services.async_call(
            "climate", "set_temperature", data, context=self._context
        )

    async def async_set_preset_mode(self, preset_mode: str) -> None:
        """Set new preset mode."""
        if preset_mode not in (self.preset_modes or []):
            raise ValueError(
                f"Got unsupported preset_mode {preset_mode}. Must be one of"
                f" {self.preset_modes}"
            )
        if preset_mode == self._attr_preset_mode:
            # I don't think we need to call async_write_ha_state if we didn't change the state
            return
        if preset_mode == PRESET_NONE:
            self._attr_preset_mode = PRESET_NONE
            self._target_temp = self._saved_target_temp
            await self._async_control_heating(force=True)
        else:
            if self._attr_preset_mode == PRESET_NONE:
                self._saved_target_temp = self._target_temp
            self._attr_preset_mode = preset_mode
            self._target_temp = self._presets[preset_mode]
            await self._async_control_heating(force=True)

        self.async_write_ha_state()

"""Atmosphere model: ISA troposphere with a temperature offset.

Valid from -500 m to 11 km (troposphere), dry air.
Pressure follows the standard ISA profile; the temperature offset dT
changes temperature (hence density and speed of sound) but not pressure.
This is the usual 'ISA + dT' convention used in firing tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T0 = 288.15            # K, sea-level standard temperature
P0 = 101_325.0         # Pa, sea-level standard pressure
LAPSE = 0.0065         # K/m, tropospheric lapse rate
G0 = 9.80665           # m/s^2
R_AIR = 287.05287      # J/(kg K), specific gas constant of dry air
GAMMA = 1.4            # ratio of specific heats
RHO0 = P0 / (R_AIR * T0)
EXPONENT = G0 / (R_AIR * LAPSE)    # ~5.256
H_MIN, H_MAX = -500.0, 11_000.0


def _alt(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < H_MIN) or np.any(h > H_MAX):
        raise ValueError(f"altitude must be within [{H_MIN}, {H_MAX}] m")
    return h


def isa_temperature(h, dT=0.0):
    """Air temperature [K] at altitude h [m MSL] with offset dT [K]."""
    return T0 - LAPSE * _alt(h) + dT


def isa_pressure(h):
    """Static pressure [Pa] at altitude h [m MSL] (standard profile)."""
    return P0 * ((T0 - LAPSE * _alt(h)) / T0) ** EXPONENT


def air_density(h, dT=0.0):
    """Air density [kg/m^3] from the ideal gas law."""
    return isa_pressure(h) / (R_AIR * isa_temperature(h, dT))


def speed_of_sound(h, dT=0.0):
    """Speed of sound [m/s] in dry air."""
    return np.sqrt(GAMMA * R_AIR * isa_temperature(h, dT))


def density_altitude(rho):
    """ISA altitude [m] that has the given density."""
    ratio = np.asarray(rho, dtype=float) / RHO0
    return (T0 / LAPSE) * (1.0 - ratio ** (1.0 / (EXPONENT - 1.0)))


@dataclass(frozen=True)
class Atmosphere:
    """Atmosphere seen from a weapon site.

    site_alt : weapon altitude above mean sea level [m]
    dT       : temperature offset from ISA [K]
    Methods take z_up, the ENU height above the weapon [m].
    """

    site_alt: float = 0.0
    dT: float = 0.0

    def _h(self, z_up):
        return self.site_alt + np.asarray(z_up, dtype=float)

    def temperature(self, z_up=0.0):
        return isa_temperature(self._h(z_up), self.dT)

    def pressure(self, z_up=0.0):
        return isa_pressure(self._h(z_up))

    def density(self, z_up=0.0):
        return air_density(self._h(z_up), self.dT)

    def sound_speed(self, z_up=0.0):
        return speed_of_sound(self._h(z_up), self.dT)
from poliastro.constants import GM_earth, R_earth
from math import sqrt

EARTH_RADII = R_earth.value * 1e-3  # Convert to km
EARTH_SURFACE_VELOCITY = sqrt((GM_earth * 1e-9) / (EARTH_RADII))  # Convert to km

#!/usr/bin/env python
import numpy as np
import math

def pixels_in(box):
    """
    Calculate the total number of pixels to be queried.
    """
    r_width = (box[2] - box[0])
    if r_width == 0:
        r_width = 1  # either a point or line
    r_height = (box[3] - box[1])
    if r_height == 0:
        r_height = 1  # either a point or line
    return r_width * r_height


def toECEF(lat, lon, alt):
    """
    Converts lat,lon,alt to Euclidean x,y,z. lat and lon must be given in
    radians, alt in metres.  NB this function is only accurate for Perth,
    Western Australia!

    alt is provided with respect to the EGM96 geoid. Therefore it needs to be
    converted to height above the WGS 84 ellipsoid at the given latitude,
    longitude. Therefore: h = alt + (EGM96 geoid height at lat lon) - (WGS84
    height at lat lon) We set the discrepancy here as being local to Western
    Australia. More complicated code would have to use a lookup table and a
    database to retrieve the correct difference based on location.
    """
    # Semi-major axis (equatorial) in metres by convension of WGS84 ellipsoid
    a = 6378137.0
    # Flattening ((a-b)/a) by convension of WGS84 ellipsoid
    f = 1.0 / 298.257223563
    e2 = 2 * f - f * f
    v = a / math.sqrt(1 - e2 * math.pow(math.sin(lat), 2))
    go = -32.6907
    h = alt - go
    x = (v + h) * math.cos(lat) * math.cos(lon)
    y = (v + h) * math.cos(lat) * math.sin(lon)
    z = ((1 - e2) * v + h) * math.sin(lat)
    return np.array([x, y, z])

import numpy as np
from datetime import datetime
from sgp4.api import Satrec
from sgp4.propagation import sgp4
from nearmiss.utils._astro._constants import (
    EARTH_RADII,
    EARTH_SURFACE_VELOCITY,
)
from nearmiss.utils._astro._dataclasses import SingleSatInputAttributes


def satellite_attributes_from_Satrec_obj(
    sat: Satrec, D_start: datetime, tle_epoch: datetime
) -> SingleSatInputAttributes:
    satnum = sat.satnum
    ndot = sat.ndot
    nddot = sat.nddot
    bstar = sat.bstar
    inclo = sat.inclo
    nodeo = sat.nodeo
    ecco = sat.ecco
    argpo = sat.argpo
    mo = sat.mo
    no_kozai = sat.no_kozai
    a = sat.a
    altp = sat.altp
    alta = sat.alta
    argpdot = sat.argpdot
    mdot = sat.mdot
    nodedot = sat.nodedot
    am = sat.am
    em = sat.em
    im = sat.im
    Om = sat.Om
    om = sat.om
    mm = sat.mm
    nm = sat.nm

    jd = np.atleast_1d(sat.jdsatepoch)
    fr = np.atleast_1d(sat.jdsatepochF)
    _, r, v = sat.sgp4_array(jd, fr)

    r_x = r[0][0] / EARTH_RADII
    r_y = r[0][1] / EARTH_RADII
    r_z = r[0][2] / EARTH_RADII

    v_x = v[0][0] / EARTH_SURFACE_VELOCITY
    v_y = v[0][1] / EARTH_SURFACE_VELOCITY
    v_z = v[0][2] / EARTH_SURFACE_VELOCITY

    tle_age = ((D_start - tle_epoch).total_seconds()) / 3600

    MLInputAttri = SingleSatInputAttributes(
        satnum,
        ndot,
        nddot,
        bstar,
        inclo,
        nodeo,
        ecco,
        argpo,
        mo,
        no_kozai,
        a,
        altp,
        alta,
        argpdot,
        mdot,
        nodedot,
        am,
        em,
        im,
        Om,
        om,
        mm,
        nm,
        r_x,
        r_y,
        r_z,
        v_x,
        v_y,
        v_z,
        tle_age,
    )
    return MLInputAttri

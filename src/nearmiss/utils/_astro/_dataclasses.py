from dataclasses import dataclass


@dataclass
class SingleSatInputAttributes:
    """
    Dataclass representing the input attributes for the ML model.

    Attributes
    ----------
    satnum : int
        Catalog number of the satellite. DO NOT include it for training the ML model.
    ndot : float
        First time derivative of mean motion.
    nddot : float
        Second time derivative of mean motion.
    bstar : float
        Ballistic drag coefficient.
    inclo : float
        Inclination in radians.
    nodeo : float
        Right Ascension of Ascending Node in radians.
    ecco : float
        Eccentricity.
    argpo : float
        Argument of perigee in radians.
    mo : float
        Mean anomaly in radians.
    no_kozai : float
        Mean motion in radians per minute.
    a : float
        Semi-major axis in Earth radii.
    altp : float
        Altitude of the satellite at perigee in Earth radii.
    alta : float
        Altitude of the satellite at apogee in Earth radii (assuming a spherical Earth).
    argpdot : float
        Rate of change of the argument of perigee in radians per minute.
    mdot : float
        Rate of change of the mean anomaly in radians per minute.
    nodedot : float
        Rate of change of the right ascension of the ascending node in radians per minute.
    am : float
        Average semi-major axis in Earth radii.
    em : float
        Average eccentricity.
    im : float
        Average inclination in radians.
    Om : float
        Average right ascension of the ascending node in radians.
    om : float
        Average argument of perigee in radians.
    mm : float
        Average mean anomaly in radians.
    nm : float
        Average mean motion in radians per minute.
    r_x : float
        X component of the position vector in the ECI frame (in Earth radii).
    r_y : float
        Y component of the position vector in the ECI frame (in Earth radii).
    r_z : float
        Z component of the position vector in the ECI frame (in Earth radii).
    v_x : float
        X component of the velocity vector in the ECI frame (in sqrt(GM_earth/R_earth)).
    v_y : float
        Y component of the velocity vector in the ECI frame (in sqrt(GM_earth/R_earth)).
    v_z : float
        Z component of the velocity vector in the ECI frame (in sqrt(GM_earth/R_earth)).
    tle_age : float
        TLE age in hours, calculated as (D_start - TLE_epoch).total_seconds() / 3600.
    """

    satnum: (
        int  # Catlog number of satellite. DO NOT include it for training of ML model.
    )
    ndot: float  # First time derivative of mean motion.
    nddot: float  # second time derivative of mean motion.
    bstar: float  # Ballastic drag coefficient.
    inclo: float  # Inclination in radians.
    nodeo: float  # Right Ascension of Ascending Node in radians.
    ecco: float  # Eccentricity.
    argpo: float  # Argument of perigee in radians.
    mo: float  # Mean anomaly in radians.
    no_kozai: float  # Mean motion radians/min.
    a: float  # Semi-major axis (earth radii).
    altp: float  # Altitude of the satellite at perigee.
    alta: float  # Altitude of the satellite at apogee (earth radii, assuming a spherical Earth).
    argpdot: (
        float  # Rate at which the argument of perigee is changing (radians/minute).
    )
    mdot: float  # Rate at which the mean anomaly is changing (radians/minute).
    nodedot: float  # Rate at which the right ascension of the ascending node is changing (radians/minute).
    am: float  # Average semi-major axis (earth radii).
    em: float  # Average eccentricity.
    im: float  # Average inclination (radians).
    Om: float  # Average right ascension of ascending node (radians).
    om: float  # Average argument of perigee (radians).
    mm: float  # Average mean anomaly (radians).
    nm: float  # Average mean motion (radians/minute).
    r_x: float  # X component of position vector in ECI frame/earth_radii
    r_y: float  # Y component of position vector in ECI frame/earth_radii
    r_z: float  # Z component of position vector in ECI frame/earth_radii
    v_x: float  # X component of velocity vector in ECI frame/sqrt(GM_earth/R_earth)
    v_y: float  # Y component of velocity vector in ECI frame/sqrt(GM_earth/R_earth)
    v_z: float  # Z component of velocity vector in ECI frame/sqrt(GM_earth/R_earth)
    tle_age: float  # TLE age. (D_start - TLE_epoch).total_seconds()/3600


@dataclass
class MLOutputAttributes:
    """
    Dataclass representing the output attributes for the ML model.

    Attributes
    ----------
    filter_rej_code : int
        Integer code indicating which filter rejected the pair (0 if no rejection).
    t_close : float
        Time of closest approach in the J2000 system, scaled by 1e5.
    ln_d_min : float
        Natural logarithm of (1 + minimum distance).
    probab : float
        Probability of collision at the closest approach.
    """

    filter_rej_code: int  # Integer code indicating which filter rejected the pair (0 if no rejection).
    t_close: float  # Time of closest approach in J2000 system/1e5
    ln_d_min: float  # ln(1 + minimum_distance)
    probab: float  # Probability of collision at closest approach.


@dataclass
class SatPairAttributes:
    """
    Dataclass representing the input attributes of pair of satellites and their outputs.

    Attributes
    ----------
    sat_1_inp : SingleSatInputAttributes
        Input attributes of satellite 1.
    sat_2_inp : SingleSatInputAttributes
        Input attributes of satellite 2.
    output : MLOutputAttributes
        Output attributes of pair of satellites.
    """

    sat_1_inp: SingleSatInputAttributes
    sat_2_inp: SingleSatInputAttributes
    output: MLOutputAttributes

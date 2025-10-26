import numpy as np
from datetime import datetime
from nearmiss.utils._astro import propagate_sgp4
from sgp4.api import Satrec


def apoapsis_periapsis_filter(
    r_1_p: float, r_2_p: float, r_1_a: float, r_2_a: float, Dist: float
) -> bool:
    if (max(r_1_p, r_2_p) - min(r_1_a, r_2_a)) > Dist:
        return True
    return False


def bounding_box_filter(
    sat1: Satrec,
    sat2: Satrec,
    epoch: datetime,
    D_start: datetime,
    D_stop: datetime,
    dt_coarse: float = 300,
) -> bool:
    """
    Returns True if bounding boxes of sat1 and sat2 overlap, else False.
    dt_coarse: coarse propagation step in seconds (default 5 minutes).
    """
    # Number of steps
    total_sec = (D_stop - D_start).total_seconds()
    n_steps = int(np.ceil(total_sec / dt_coarse)) + 1
    times = np.linspace(0, total_sec, n_steps)

    # Precompute positions for sat1
    positions1 = np.array([propagate_sgp4(sat1, t, epoch)[0] for t in times])
    min1 = positions1.min(axis=0)
    max1 = positions1.max(axis=0)

    # Precompute positions for sat2
    positions2 = np.array([propagate_sgp4(sat2, t, epoch)[0] for t in times])
    min2 = positions2.min(axis=0)
    max2 = positions2.max(axis=0)

    # Only skip if there is NO overlap on any axis
    if (
        max1[0] < min2[0]
        or min1[0] > max2[0]
        or max1[1] < min2[1]
        or min1[1] > max2[1]
        or max1[2] < min2[2]
        or min1[2] > max2[2]
    ):
        return False  # No overlap -> skip
    else:
        return True  # Overlap exists -> potential close approach


def relative_motion_filter(
    r1: np.ndarray,
    v1: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    window_seconds: float,
    threshold_km: float,
    safety_margin_km: float = 0.0,
) -> bool:
    """
    Fast relative-motion / bounding-line filter.

    Uses a linear relative-motion approximation to find the minimum distance
    between two objects within a time window of length `window_seconds`.
    If the minimum distance is larger than threshold_km + safety_margin_km,
    the function returns False (pair can be safely skipped). Otherwise returns True.

    Parameters
    ----------
    r1, v1 : np.ndarray
        Position (km) and velocity (km/s) of object 1 at the reference epoch.
    r2, v2 : np.ndarray
        Position (km) and velocity (km/s) of object 2 at the reference epoch.
    window_seconds : float
        Duration (seconds) of the time window to search for possible close approach.
        The linear approximation assumes window_seconds is not too large (e.g. < 1 orbit).
    threshold_km : float
        Distance threshold (km). If the predicted min distance > threshold_km + safety_margin_km,
        the pair is rejected (returns False).
    safety_margin_km : float, optional
        Extra margin (km) to be conservative. Default 0.0.

    Returns
    -------
    bool
        True if the pair should be examined further (i.e., it MAY come within threshold),
        False if it can be skipped (i.e., predicted min distance > threshold + margin).
    """
    # Basic input checks
    if r1.shape != (3,) or r2.shape != (3,) or v1.shape != (3,) or v2.shape != (3,):
        raise ValueError("r1, r2, v1, v2 must be 1D numpy arrays with shape (3,)")

    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    if threshold_km <= 0:
        raise ValueError("threshold_km must be positive")

    # Relative vectors
    r0 = r2 - r1  # km
    vrel = v2 - v1  # km/s

    vrel_sq = np.dot(vrel, vrel)
    # If relative velocity is (near-)zero then treat as static
    if vrel_sq < 1e-12:
        # distance is static for the window
        d_static = np.linalg.norm(r0)
        return d_static <= (threshold_km + safety_margin_km)

    # continuous-time minimizer along straight relative motion
    t_star = -np.dot(r0, vrel) / vrel_sq

    # clamp to window [0, window_seconds]
    t_clamped = min(max(t_star, 0.0), float(window_seconds))

    # Evaluate squared distances at t_clamped and at endpoints 0 and window_seconds
    def dist2_at(t):
        rr = r0 + vrel * t
        return np.dot(rr, rr)

    d2_candidates = (
        dist2_at(0.0),
        dist2_at(float(window_seconds)),
        dist2_at(t_clamped),
    )
    dmin = np.sqrt(min(d2_candidates))

    # final decision
    if dmin > (threshold_km + safety_margin_km):
        # definitely safe -> skip
        return False
    else:
        # possible close approach -> keep for detailed analysis
        return True


def _angular_difference(deg1: float, deg2: float) -> float:
    """Compute smallest angular difference between two angles in degrees."""
    diff = abs(deg1 - deg2) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff


def inclination_raan_filter(
    inc1_deg: float,
    inc2_deg: float,
    raan1_deg: float,
    raan2_deg: float,
    inc_tol_deg: float = 10.0,
    raan_tol_deg: float = 15.0,
) -> bool:
    """
    Quick geometric filter based on inclination and RAAN.
    Returns True if the two orbital planes are close enough that a conjunction
    is geometrically possible. Returns False if planes are too far apart.

    Parameters
    ----------
    inc1_deg, inc2_deg : float
        Inclinations of the two orbits in degrees.
    raan1_deg, raan2_deg : float
        RAANs of the two orbits in degrees.
    inc_tol_deg : float, optional
        Maximum allowed difference in inclination. Default = 10 deg.
    raan_tol_deg : float, optional
        Maximum allowed difference in RAAN. Default = 15 deg.

    Returns
    -------
    bool
        True if the pair could potentially intersect geometrically.
        False if orbital planes are too misaligned to ever meet.
    """
    d_inc = _angular_difference(inc1_deg, inc2_deg)
    d_raan = _angular_difference(raan1_deg, raan2_deg)

    # planes nearly opposite (retrograde vs prograde)? reject
    if abs(inc1_deg - inc2_deg) > 90:
        return False

    # both inclination and RAAN must be within tolerance to pass
    if (d_inc <= inc_tol_deg) and (d_raan <= raan_tol_deg):
        return True
    return False

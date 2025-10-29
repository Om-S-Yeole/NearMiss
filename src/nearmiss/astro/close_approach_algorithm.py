"""
Module for determining the closest approach and collision probability between objects.

This module provides a function to calculate the closest approach and collision probability between two objects based on their position, velocity, and covariance data.
"""

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from datetime import datetime, timedelta
from math import sqrt, isclose
from filterpy.kalman import MerweScaledSigmaPoints, unscented_transform
from astropy import units as u
from astropy.time import TimeDelta
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import ValladoPropagator
from nearmiss.utils import (
    RSW_to_ECI_covariance,
    cov_matrix_from_ECI_to_NTW_frame_converter,
    acc_vec_calculator,
    cubic_spline_root_finder,
    quintic_polynomial_maker,
    t_from_tau,
    semi_major_minor_axis_from_cov_NTW,
    ellipsoidal_function,
    four_point_cubic_spline_root_finder,
    max_prob_function,
)


def close_approach_physical_algorithm(
    r_p: np.ndarray,
    v_p: np.ndarray,
    r_s: np.ndarray,
    v_s: np.ndarray,
    t_0: datetime,
    D_start: datetime,
    D_stop: datetime,
    Del_t: float = 3600.0,
    Dist: float = 10.0,
    initial_cov_p: np.ndarray = np.diag([1, 1, 1, 4e-4, 4e-4, 4e-4]),
    initial_cov_s: np.ndarray = np.diag([1, 1, 1, 4e-4, 4e-4, 4e-4]),
    initial_cov_in_RSW: bool = False,
    r_obj_1: float = 1.5,
    r_obj_2: float = 1.5,
) -> tuple[datetime | None, float | None, float]:
    """
    Determine the closest approach and collision probability between two objects.

    Parameters
    ----------
    r_p : np.ndarray
        Position vector of the primary object in km.
    v_p : np.ndarray
        Velocity vector of the primary object in km/s.
    r_s : np.ndarray
        Position vector of the secondary object in km.
    v_s : np.ndarray
        Velocity vector of the secondary object in km/s.
    t_0 : datetime
        Initial time of the state vectors.
    D_start : datetime
        Start time of the analysis window.
    D_stop : datetime
        End time of the analysis window.
    Del_t : float, optional
        Time step in seconds. Default is 3600.0 seconds.
    Dist : float, optional
        Minimum distance threshold for collision in km. Default is 10.0 km.
    initial_cov_p : np.ndarray, optional
        Initial covariance matrix for the primary object. Default is a diagonal matrix.
    initial_cov_s : np.ndarray, optional
        Initial covariance matrix for the secondary object. Default is a diagonal matrix.
    initial_cov_in_RSW : bool, optional
        Whether the initial covariance is in RSW frame. Default is False.
    r_obj_1 : float, optional
        Radius of the primary object in meters. Default is 1.5 m.
    r_obj_2 : float, optional
        Radius of the secondary object in meters. Default is 1.5 m.

    Returns
    -------
    tuple[datetime | None, float | None, float]
        A tuple containing:
        - Time of closest approach (datetime or None).
        - Closest distance (float or None).
        - Maximum collision probability (float).

    Raises
    ------
    TypeError
        If input parameters are of incorrect types.
    ValueError
        If input parameters are invalid or inconsistent.

    Notes
    -----
    - The function uses orbital propagation to determine the closest approach.
    - Covariance matrices are used to calculate collision probabilities.
    """

    # Type validation
    if not isinstance(r_p, np.ndarray):
        raise TypeError(f"Expected r_p to be np.ndarray, got {type(r_p)}.")
    if not isinstance(v_p, np.ndarray):
        raise TypeError(f"Expected v_p to be np.ndarray, got {type(v_p)}.")
    if not isinstance(r_s, np.ndarray):
        raise TypeError(f"Expected r_s to be np.ndarray, got {type(r_s)}.")
    if not isinstance(v_s, np.ndarray):
        raise TypeError(f"Expected v_s to be np.ndarray, got {type(v_s)}.")
    if not isinstance(t_0, datetime):
        raise TypeError(f"Expected t_0 to be datetime, got {type(t_0)}.")
    if not isinstance(D_start, datetime):
        raise TypeError(f"Expected D_start to be datetime, got {type(D_start)}.")
    if not isinstance(D_stop, datetime):
        raise TypeError(f"Expected D_stop to be datetime, got {type(D_stop)}.")
    if not isinstance(Del_t, (float, int)):
        raise TypeError(f"Expected Del_t to be float or int, got {type(Del_t)}.")
    if not isinstance(Dist, (float, int)):
        raise TypeError(f"Expected Dist to be float or int, got {type(Dist)}.")
    if not isinstance(initial_cov_p, np.ndarray):
        raise TypeError(
            f"Expected initial_cov_p to be np.ndarray, got {type(initial_cov_p)}."
        )
    if not isinstance(initial_cov_s, np.ndarray):
        raise TypeError(
            f"Expected initial_cov_s to be np.ndarray, got {type(initial_cov_s)}."
        )
    if not isinstance(initial_cov_in_RSW, bool):
        raise TypeError(
            f"Expected initial_cov_in_RSW to be bool, got {type(initial_cov_in_RSW)}."
        )
    if not isinstance(r_obj_1, (float, int)):
        raise TypeError(f"Expected r_obj_1 to be float or int, got {type(r_obj_1)}.")
    if not isinstance(r_obj_2, (float, int)):
        raise TypeError(f"Expected r_obj_2 to be float or int, got {type(r_obj_2)}.")

    if D_start >= D_stop:
        raise ValueError(
            f"D_start can not be greater than or equal to D_stop. Got D_start: {D_start} and D_stop: {D_stop}"
        )
    if (D_stop - D_start) > timedelta(days=7):
        raise ValueError(
            f"Maximum time frame limit allowed to check the maximum probability of collision is 7 days. Please adjust the values of D_start and D_stop accordingly."
        )
    if t_0 >= D_stop:
        raise ValueError(
            f"Value of t_0 can not be greater than or equal to D_stop. Got t_0: {t_0} and D_stop: {D_stop}"
        )
    if (t_0 < D_start) and ((D_start - t_0) > timedelta(days=3)):
        raise ValueError(
            f"State vectors earlier than 3 days of the time frame start (D_start) time are not allowed. This leads to inaccuracy in calculations. Please provide fresh state vectors."
        )

    orb_p = Orbit.from_vectors(
        Earth, u.Quantity(r_p, unit=u.km), u.Quantity(v_p, unit=u.km / u.s)
    )
    orb_s = Orbit.from_vectors(
        Earth, u.Quantity(r_s, unit=u.km), u.Quantity(v_s, unit=u.km / u.s)
    )

    # Periapsis-Apoapsis Filter
    r_periapsis_max = max(orb_p.r_p.value, orb_s.r_p.value)
    r_apoapsis_min = min(orb_p.r_a.value, orb_s.r_a.value)
    if (r_periapsis_max - r_apoapsis_min) > Dist:
        # If difference between maximum periapsis and minimum apoapsis is greater than Dist, then there is no possibility of collision.
        print(
            f"Periapsis-Apoapsis filter activated for this scenerio. Objects will not collide. r_periapsis_max - r_apoapsis_min: {r_periapsis_max - r_apoapsis_min}"
        )
        return (None, None, 0)

    if t_0 < D_start:
        # If the state vectors in order to create the orbits of primary and secondary satellites are given at epoch which is less than D_start, then propagate those state vectors to their state at D_start and update t_0 to be equal to D_start.
        td = TimeDelta(D_start - t_0)
        orb_p = orb_p.propagate(td, method=ValladoPropagator(numiter=800))
        orb_s = orb_s.propagate(td, method=ValladoPropagator(numiter=800))
        t_0 = D_start

    state_vector_dim = 6
    initial_state_vector_p = np.concatenate((r_p, v_p))
    initial_state_vector_s = np.concatenate((r_s, v_s))

    scaled_sigma_points = MerweScaledSigmaPoints(
        n=state_vector_dim, alpha=0.1, beta=2.0, kappa=3 - state_vector_dim
    )

    if initial_cov_in_RSW:
        initial_cov_p = RSW_to_ECI_covariance(r_p, v_p, initial_cov_p)
        initial_cov_s = RSW_to_ECI_covariance(r_s, v_s, initial_cov_s)

    state_samples_p = scaled_sigma_points.sigma_points(
        initial_state_vector_p, initial_cov_p
    )
    state_samples_s = scaled_sigma_points.sigma_points(
        initial_state_vector_s, initial_cov_s
    )

    Wm_p = Wm_s = scaled_sigma_points.Wm
    Wc_p = Wc_s = scaled_sigma_points.Wc

    del_t = timedelta(seconds=Del_t)
    del_t_as = TimeDelta(del_t)

    # From these list, return the t_close, d_close, and P_max when d_close is minimum.
    t_close_list: list[datetime] = []
    d_close_list: list[float] = []
    P_max_list: list[float] = []

    while t_0 < D_stop:
        orb_p_propagated = orb_p.propagate(
            del_t_as, method=ValladoPropagator(numiter=800)
        )
        orb_s_propagated = orb_s.propagate(
            del_t_as, method=ValladoPropagator(numiter=800)
        )

        new_state_sigmas_propagated_p = []
        new_state_sigmas_propagated_s = []

        for state in state_samples_p:
            orb_p_sampled = Orbit.from_vectors(
                Earth,
                u.Quantity(state[:3], u.km),
                u.Quantity(state[3:], unit=u.km / u.s),
            )
            orb_p_sampled_prop = orb_p_sampled.propagate(
                del_t_as, method=ValladoPropagator(numiter=800)
            )
            new_state_sample_p = np.concatenate(
                (orb_p_sampled_prop.r.value, orb_p_sampled_prop.v.value)
            )
            new_state_sigmas_propagated_p.append(new_state_sample_p)

        for state in state_samples_s:
            orb_s_sampled = Orbit.from_vectors(
                Earth,
                u.Quantity(state[:3], u.km),
                u.Quantity(state[3:], unit=u.km / u.s),
            )
            orb_s_sampled_prop = orb_s_sampled.propagate(
                del_t_as, method=ValladoPropagator(numiter=800)
            )
            new_state_sample_s = np.concatenate(
                (orb_s_sampled_prop.r.value, orb_s_sampled_prop.v.value)
            )
            new_state_sigmas_propagated_s.append(new_state_sample_s)

        new_state_sigmas_propagated_p = np.array(new_state_sigmas_propagated_p)
        new_state_sigmas_propagated_s = np.array(new_state_sigmas_propagated_s)

        mean_predicted_p_ECI, cov_predicted_p_ECI = unscented_transform(
            new_state_sigmas_propagated_p, Wm_p, Wc_p
        )
        mean_predicted_s_ECI, cov_predicted_s_ECI = unscented_transform(
            new_state_sigmas_propagated_s, Wm_s, Wc_s
        )

        cov_predicted_p_NTW = cov_matrix_from_ECI_to_NTW_frame_converter(
            r=orb_p_propagated.r.value,
            v=orb_p_propagated.v.value,
            cov=cov_predicted_p_ECI,
        )
        cov_predicted_s_NTW = cov_matrix_from_ECI_to_NTW_frame_converter(
            r=orb_s_propagated.r.value,
            v=orb_s_propagated.v.value,
            cov=cov_predicted_s_ECI,
        )
        new_combined_cov_NTW = cov_predicted_p_NTW + cov_predicted_s_NTW

        r_d_prev = orb_s.r.value - orb_p.r.value
        v_d_prev = orb_s.v.value - orb_p.v.value
        acc_d_prev = acc_vec_calculator(orb_s.r.value) - acc_vec_calculator(
            orb_p.r.value
        )

        r_d_after = orb_s_propagated.r.value - orb_p_propagated.r.value
        v_d_after = orb_s_propagated.v.value - orb_p_propagated.v.value
        acc_d_after = acc_vec_calculator(orb_s_propagated.r.value) - acc_vec_calculator(
            orb_p_propagated.r.value
        )

        # g_d_prev = np.dot(r_d_prev, r_d_prev) # Not used further
        g_dot_d_prev = 2 * (np.dot(v_d_prev, r_d_prev))
        g_dou_dot_d_prev = 2 * (
            np.dot(acc_d_prev, r_d_prev) + np.dot(v_d_prev, v_d_prev)
        )

        # g_d_after = np.dot(r_d_after, r_d_after) # Not used further
        g_dot_d_after = 2 * (np.dot(v_d_after, r_d_after))
        g_dou_dot_d_after = 2 * (
            np.dot(acc_d_after, r_d_after) + np.dot(v_d_after, v_d_after)
        )

        cubic_spline_real_roots = {
            np.real(root)
            for root in cubic_spline_root_finder(
                g_dot_d_prev, g_dot_d_after, g_dou_dot_d_prev, g_dou_dot_d_after, Del_t
            )
            if np.isreal(root)
        }

        if not cubic_spline_real_roots:
            print("Unable to get the real root from cubic spline.")
            return (None, None, 0)

        tau_d_root: float | None = None

        for root in cubic_spline_real_roots:
            dt = TimeDelta(timedelta(seconds=root * Del_t))
            orb_p_temp_prop = orb_p.propagate(dt, method=ValladoPropagator(numiter=800))
            orb_s_temp_prop = orb_s.propagate(dt, method=ValladoPropagator(numiter=800))
            r_d_after_temp = orb_s_temp_prop.r.value - orb_p_temp_prop.r.value
            v_d_after_temp = orb_s_temp_prop.v.value - orb_p_temp_prop.v.value
            acc_d_after_temp = acc_vec_calculator(
                orb_s_temp_prop.r.value
            ) - acc_vec_calculator(orb_p_temp_prop.r.value)

            g_dot_d_temp = 2 * (np.dot(v_d_after_temp, r_d_after_temp))
            g_dou_dot_d_temp = 2 * (
                np.dot(acc_d_after_temp, r_d_after_temp)
                + np.dot(v_d_after_temp, v_d_after_temp)
            )

            if isclose(g_dot_d_temp, 0.0, abs_tol=1e-5) and g_dou_dot_d_temp > 0:
                tau_d_root = root
                break

        if not tau_d_root:
            print(
                f"Unable to find an optimal value for tau_d_root for {t_0} to {t_0 + del_t}. Continue to next iteration."
            )
            t_0 = (
                t_0 + del_t
            )  # Update the current time -> Necessary step for while loop to stop
            orb_p = orb_p_propagated
            orb_s = orb_s_propagated
            state_samples_p = scaled_sigma_points.sigma_points(
                x=mean_predicted_p_ECI, P=cov_predicted_p_ECI
            )
            state_samples_s = scaled_sigma_points.sigma_points(
                x=mean_predicted_s_ECI, P=cov_predicted_s_ECI
            )
            continue

        P_qi_list: list[Polynomial] = []
        d_close: float | None = None

        for i in range(3):
            parameter_dict = {
                "f_n": r_d_prev[i],
                "f_n_1": r_d_after[i],
                "f_dot_n": v_d_prev[i],
                "f_dot_n_1": v_d_after[i],
                "f_dou_dot_n": acc_d_prev[i],
                "f_dou_dot_n_1": acc_d_after[i],
                "del_t": Del_t,
            }
            P_qi_list.append(quintic_polynomial_maker(**parameter_dict))

        P_qi_I, P_qi_J, P_qi_K = P_qi_list

        d_close = sqrt(
            P_qi_I.__call__(tau_d_root) ** 2
            + P_qi_J.__call__(tau_d_root) ** 2
            + P_qi_K.__call__(tau_d_root) ** 2
        )
        t_close: float = t_from_tau(t_0, tau_d_root, Del_t)

        a, b = semi_major_minor_axis_from_cov_NTW(new_combined_cov_NTW)

        tau_list = [0, 1 / 3, 2 / 3, 1]
        p_list = []

        for tau in tau_list:
            td = TimeDelta(timedelta(tau * Del_t))
            orb_p_ellip_prop = orb_p.propagate(
                td, method=ValladoPropagator(numiter=800)
            )
            orb_s_ellip_prop = orb_s.propagate(
                td, method=ValladoPropagator(numiter=800)
            )
            r_d_ellip = orb_s_ellip_prop.r.value - orb_p_ellip_prop.r.value
            v_p_ellip = orb_p_ellip_prop.v.value
            p_list.append(ellipsoidal_function(r_d_ellip, v_p_ellip, a, b))

        four_point_cubic_spline_real_roots = [
            np.real(root)
            for root in four_point_cubic_spline_root_finder(
                np.array(tau_list), np.array(p_list)
            )
            if np.isreal(root)
        ]

        if four_point_cubic_spline_real_roots:
            # Convert the satellite radii from m to km.
            P_max = max_prob_function(r_obj_1 / 1000, r_obj_2 / 1000, d_close)
            d_close_list.append(d_close)
            t_close_list.append(t_close)
            P_max_list.append(P_max)

        t_0 = (
            t_0 + del_t
        )  # Update the current time -> Necessary step for while loop to stop
        orb_p = orb_p_propagated
        orb_s = orb_s_propagated
        state_samples_p = scaled_sigma_points.sigma_points(
            x=mean_predicted_p_ECI, P=cov_predicted_p_ECI
        )
        state_samples_s = scaled_sigma_points.sigma_points(
            x=mean_predicted_s_ECI, P=cov_predicted_s_ECI
        )

    if not d_close_list:
        print("No valid close-approach minimum was found in the specified time window.")
        return (None, None, 0)

    d_min = min(d_close_list)
    d_min_idx = d_close_list.index(d_min)
    t_close_at_d_min = t_close_list[d_min_idx]
    P_max_at_d_min = P_max_list[d_min_idx]

    return (t_close_at_d_min, d_min, P_max_at_d_min)

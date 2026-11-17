"""Associate ball candidates against an adaptive, camera-compensated local track.

Only accepted image observations update velocity and acceleration uncertainty.
Predictions never train their own motion model. Loss increases uncertainty for
a bounded horizon, after which the estimate is explicitly stale. A fresh track
requires a short, consistent run of image observations.
"""
import copy

import numpy as np

from recover_blank_tracks import candidates, trusted


def seed_observation(rows, index, limits):
    """Require three nearby image observations before initializing motion."""
    if index + 2 >= len(rows):
        return None
    points = []
    pose = np.eye(3)
    for j in range(index, index+3):
        if j > index:
            affine = rows[j].get('camera_affine')
            if affine is None or rows[j]['frame'] != rows[j-1]['frame']+1:
                return None
            transform = np.eye(3)
            transform[:2] = affine
            pose = transform @ pose
        point = trusted(rows[j])
        if point is None or (rows[j].get('penalty_spot_guard') or {}).get('rejected'):
            return None
        points.append(np.linalg.solve(pose, np.r_[point, 1])[:2])
    velocity = np.diff(points, axis=0)
    limit = limits[index]
    if (np.linalg.norm(velocity, axis=1).max() > limit['max_speed_px_frame']
            or np.linalg.norm(velocity[1]-velocity[0]) > limit['max_acceleration_px_frame2'] + 6):
        return None
    return points[0], velocity.mean(axis=0)


def filter_local_motion(rows, limits, zoom, candidate_allowed, people=None, horizon=50):
    """Kalman innovation gate, with process noise estimated from recent motion.

    A 99%-style 2D Mahalanobis gate is used as a heuristic, since detector errors
    are not known to be Gaussian. Its radius has a speed-dependent cap, preventing
    an occlusion from making the entire frame reachable. No rejected candidate
    updates the state. The absolute motion envelope is an additional check.
    """
    result = copy.deepcopy(rows)
    state = covariance = None
    missed = 0
    acceleration_history = []
    observation_history = []
    previous_observed_velocity = None
    for i, row in enumerate(result):
        scale = zoom[i]
        measurement_variance = (3*scale)**2
        affine = row.get('camera_affine')
        cut = (i == 0 or affine is None or rows[i]['frame'] != rows[i-1]['frame']+1)
        if cut:
            state = None
        if state is None:
            seed = seed_observation(rows, i, limits)
            if seed is None:
                row['context_position'] = None
                row['context_source'] = 'local_motion_unlocalized'
                row['local_motion'] = dict(state='unlocalized', accepted=False)
                continue
            point, velocity = seed
            state = np.r_[point, velocity]
            covariance = np.diag([measurement_variance]*2 + [measurement_variance/4]*2)
            missed = 0
            acceleration_history = []
            observation_history = [(point.copy(), 0)]
            previous_observed_velocity = None
            row['context_position'] = point.tolist()
            row['context_source'] = 'local_motion_observed'
            row['local_motion'] = dict(state='initialized', accepted=True,
                                       speed_px_frame=float(np.linalg.norm(velocity)))
            continue
        # Transform both state and covariance into the current image first.
        linear = np.asarray(affine, float)[:, :2]
        camera = np.zeros((4, 4))
        camera[:2, :2] = camera[2:, 2:] = linear
        state = camera @ state
        state[:2] += np.asarray(affine)[:, 2]
        covariance = camera @ covariance @ camera.T
        observation_history = [(linear@point+np.asarray(affine)[:, 2], age+1)
                               for point, age in observation_history]
        if previous_observed_velocity is not None:
            previous_observed_velocity = linear @ previous_observed_velocity
        origin = state[:2].copy()
        transition = np.eye(4)
        transition[:2, 2:] = np.eye(2)
        state = transition @ state
        predicted_position = state[:2].copy()
        local_acceleration = float(np.median(acceleration_history)) if acceleration_history else .15*scale
        process_sigma = np.clip(local_acceleration, .1*scale, limits[i]['max_acceleration_px_frame2'])
        noise_map = np.vstack((np.eye(2)*.5, np.eye(2)))
        covariance = transition @ covariance @ transition.T + process_sigma**2 * noise_map @ noise_map.T
        innovation_covariance = covariance[:2, :2] + np.eye(2)*measurement_variance
        inverse = np.linalg.inv(innovation_covariance)
        speed = float(np.linalg.norm(state[2:]))
        radius = min(float(np.sqrt(9.21*np.linalg.eigvalsh(innovation_covariance).max())),
                     12*scale + 2*speed + 4*local_acceleration)
        pool = candidates(rows[i])
        observed = trusted(rows[i])
        if observed is not None:
            pool.append([*observed, 1.])
        scored = []
        for candidate in pool:
            point = np.asarray(candidate[:2], float)
            residual = point-state[:2]
            score = float(residual @ inverse @ residual)
            if (score <= 9.21 and np.linalg.norm(residual) <= radius
                    and np.linalg.norm(point-origin) <= limits[i]['max_speed_px_frame'] + 3*scale
                    and candidate_allowed(i, point)):
                scored.append((score, point))
        accepted = bool(scored)
        target = min(scored, key=lambda item: item[0])[1] if accepted else None
        reacquired = False
        if missed > horizon:
            # A fresh coherent moving track can recover from an old wrong lock.
            # Stationary distant clutter needs nearby-player support before it
            # can replace either a stationary track or a continuing flight.
            seed = seed_observation(rows, i, limits)
            if seed is not None and candidate_allowed(i, seed[0]):
                boxes = np.asarray(people[i] if people is not None else []).reshape(-1, 4)
                feet = np.column_stack(((boxes[:, 0]+boxes[:, 2])/2, boxes[:, 3]))
                contact = len(feet) > 0 and np.linalg.norm(feet-seed[0], axis=1).min() <= 20*scale
                moving = np.linalg.norm(seed[1]) > scale
                consistent = np.linalg.norm(seed[0]-state[:2]) <= min(radius, 8*scale+2*speed)
                if moving or contact or consistent:
                    state = np.r_[seed[0], seed[1]]
                    covariance = np.diag([measurement_variance]*2 + [measurement_variance/4]*2)
                    acceleration_history = []
                    observation_history = []
                    previous_observed_velocity = None
                    target, accepted, reacquired = seed[0], True, True
                    predicted_position = seed[0].copy()
                    radius = 3*np.sqrt(measurement_variance)
                else:
                    accepted = False
            else:
                accepted = False
        if accepted:
            if not reacquired:
                gain = covariance[:, :2] @ inverse
                state += gain @ (target-state[:2])
                # Joseph update preserves positive covariance under rounding.
                correction = np.eye(4)
                correction[:, :2] -= gain
                covariance = correction @ covariance @ correction.T + measurement_variance*gain@gain.T
            if observation_history:
                point, age = observation_history[-1]
                observed_velocity = (target-point)/age
                if previous_observed_velocity is not None:
                    acceleration_history.append(float(np.linalg.norm(observed_velocity-previous_observed_velocity)/age))
                    acceleration_history = acceleration_history[-7:]
                previous_observed_velocity = observed_velocity
            observation_history.append((target.copy(), 0))
            observation_history = observation_history[-5:]
            missed = 0
        else:
            missed += 1
        row['context_position'] = state[:2].tolist()
        row['context_source'] = 'local_motion_observed' if accepted else 'local_motion_predicted'
        row['local_motion'] = dict(state='reacquired' if reacquired else 'observed' if accepted else 'stale' if missed > horizon else 'predicted', accepted=accepted,
                                   predicted_position=predicted_position.tolist(), gate_radius_px=radius,
                                   speed_px_frame=float(np.linalg.norm(state[2:])),
                                   local_acceleration_px_frame2=local_acceleration,
                                   missed_frames=missed, candidate_count=len(pool),
                                   uncertainty_radius_px=max(radius, np.sqrt(np.trace(covariance[:2, :2]))))
    return result

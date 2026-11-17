"""Penalty-spot-resistant, motion-bounded continuous display trajectory.

Original recovery evidence is retained. Unsupported penalty-spot anchors are
removed before interpolation. Display motion has explicit camera-compensated
speed and acceleration limits. Incompatible observations become flagged display
estimates, with uncertainty covering the displacement from the input position.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
from stage1_cv_baseline import private_path, digest
from local_cv_possession import atomic
from continuous_trajectory import fill
from recover_blank_tracks import finite, trusted
from sam_confirmation_recovery import read_rows
from sam2_gap_recovery import transport


CONFIG = dict(spot_radius_px=16., crossing_lookaround=8, crossing_error_px=8.,
              crossing_min_travel_px=32., contact_px=18., strong_confidence=.6,
              arrival_frames=25, max_speed_px_frame=45., max_acceleration_px_frame2=12.)


def camera_zoom_scales(rows):
    """Relative image magnification from each shot's first camera frame."""
    scales = []
    scale = 1.
    for i, row in enumerate(rows):
        affine = row.get('camera_affine')
        valid = (i > 0 and row['frame'] == rows[i-1]['frame'] + 1
                 and affine is not None and np.shape(affine) == (2, 3)
                 and np.isfinite(affine).all())
        determinant = abs(float(np.linalg.det(np.asarray(affine)[:, :2]))) if valid else 0.
        if determinant <= 1e-8:
            scale = 1.
        else:
            scale *= np.sqrt(determinant)
            if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
                raise ValueError('Camera zoom accumulation is numerically invalid')
        scales.append(float(scale))
    return scales


def scaled_config(config, scale):
    result = dict(config)
    for name in ('spot_radius_px', 'crossing_error_px', 'crossing_min_travel_px',
                 'contact_px', 'max_speed_px_frame', 'max_acceleration_px_frame2'):
        result[name] *= scale
    return result


def spots(row):
    positions = {shape['name']: np.asarray(shape['points'][0], float)
            for shape in row.get('field_geometry') or []
            if shape['name'] in ('left_penalty_spot', 'right_penalty_spot')
            and len(shape['points']) == 1 and np.isfinite(shape['points'][0]).all()}
    for name, track in (row.get('active_penalty_spots') or {}).items():
        if track.get('confirmed') and finite(track.get('center')):
            positions[name] = np.asarray(track['center'],float)
    return list(positions.items())


def nearest_spot(row, point):
    candidates = [(float(np.linalg.norm(point - center)), name, center) for name, center in spots(row)]
    return min(candidates, key=lambda item: item[0]) if candidates else (float('inf'), None, None)


def direct_confidence(row, point, radius=8.):
    values = []
    for key in ('selected', 'short_gap_detection', 'aerial_detection', 'blank_recovery_detection', 'bidirectional_detection'):
        candidate = row.get(key)
        if candidate is not None and len(candidate) == 3 and np.isfinite(candidate).all() and np.linalg.norm(np.asarray(candidate[:2]) - point) <= radius:
            values.append(candidate[2])
    return max(values, default=0.)


def same_camera(rows, first, last):
    return all(rows[i].get('camera_affine') is not None
               and rows[i]['frame'] == rows[i-1]['frame'] + 1 for i in range(first+1, last+1))


def allow_spot(rows, index, point, people, config=None):
    c = dict(CONFIG, **(config or {}))
    for name, track in (rows[index].get('active_penalty_spots') or {}).items():
        if track.get('confirmed') and finite(track.get('center')):
            distance = float(np.linalg.norm(point-np.asarray(track['center'])))
            if distance <= track['veto_radius_px']:
                return False, dict(spot=name,distance_px=distance,reason='tracked_penalty_spot',
                                   veto_radius_px=track['veto_radius_px'],tracking_state=track['state'])
    distance, name, center = nearest_spot(rows[index], point)
    if distance > c['spot_radius_px']:
        return True, dict(reason='outside_spot', distance_px=distance if name else None)
    info = dict(spot=name, distance_px=distance, reason='unsupported_penalty_spot')
    neighbors = []
    for direction in (-1, 1):
        found = None
        for dt in range(1, c['crossing_lookaround'] + 1):
            j = index + direction * dt
            if not 0 <= j < len(rows) or not same_camera(rows, min(j,index), max(j,index)):
                break
            observation = trusted(rows[j])
            if observation is not None:
                projected = transport(observation, rows, j, index)
                if np.linalg.norm(projected - center) > c['spot_radius_px'] * 1.25:
                    found = (j, projected)
                    break
        neighbors.append(found)
    if all(neighbors):
        (left,a),(right,b) = neighbors
        u = (index-left)/(right-left)
        predicted = (1-u)*a + u*b
        travel = np.linalg.norm(b-a)
        if (c['crossing_min_travel_px'] <= travel <= c['max_speed_px_frame']*(right-left)
                and np.linalg.norm(point-predicted) <= c['crossing_error_px']):
            return True, dict(info, reason='motion_supported_crossing')
    boxes = np.asarray(people[index] if people is not None else []).reshape(-1,4)
    feet = np.c_[(boxes[:,0]+boxes[:,2])/2, boxes[:,3]]
    contact = bool(len(feet)) and np.linalg.norm(feet-point,axis=1).min() <= c['contact_px']
    if contact and direct_confidence(rows[index], point, c['crossing_error_px']) >= c['strong_confidence']:
        arrivals = []
        for j in range(max(0,index-c['arrival_frames']),index):
            observed = trusted(rows[j])
            if observed is not None and same_camera(rows,j,index):
                projected = transport(observed,rows,j,index)
                arrivals.append((j,projected,float(np.linalg.norm(projected-center))))
        outside = [k for k,item in enumerate(arrivals) if item[2] > c['spot_radius_px']*1.25]
        if outside:
            path = arrivals[outside[-1]:] + [(index,point,distance)]
            if len(path) >= 3:
                inward = all(b[2] <= a[2]+c['crossing_error_px']/4 for a,b in zip(path,path[1:]))
                bounded = all(np.linalg.norm(b[1]-a[1]) <= c['max_speed_px_frame']*(b[0]-a[0]) for a,b in zip(path,path[1:]))
                if inward and bounded:
                    return True, dict(info, reason='strong_detected_arrival_with_player')
    return False, info


def clip_norm(vector, maximum):
    length = np.linalg.norm(vector)
    return vector * min(1., maximum/max(length,1e-12))


def bounded_velocity(desired, previous, speed_limit, acceleration_limit):
    """Project onto intersecting speed and acceleration disks in pixel space."""
    distance = float(np.linalg.norm(previous))
    if distance > speed_limit + acceleration_limit + 1e-9:
        raise ValueError('Camera transport makes the two motion bounds incompatible')
    speed_projection = clip_norm(desired, speed_limit)
    if np.linalg.norm(speed_projection-previous) <= acceleration_limit:
        return speed_projection
    acceleration_projection = previous + clip_norm(desired-previous, acceleration_limit)
    if np.linalg.norm(acceleration_projection) <= speed_limit:
        return acceleration_projection
    if distance < 1e-12:
        return clip_norm(desired, min(speed_limit, acceleration_limit))
    direction = previous / distance
    along = (speed_limit**2-acceleration_limit**2+distance**2)/(2*distance)
    perpendicular = np.array([-direction[1], direction[0]])
    height = np.sqrt(max(0., speed_limit**2-along**2))
    choices = [along*direction+sign*height*perpendicular for sign in (-1,1)]
    return min(choices, key=lambda v: np.linalg.norm(v-desired))


def constrain(rows, people, width, height, config=None, motion_limits=None, local_motion=False):
    c = dict(CONFIG, **(config or {}))
    scales = camera_zoom_scales(rows)
    if motion_limits is not None and len(motion_limits) != len(rows):
        raise ValueError('Motion limits must align with every frame')
    configs = [scaled_config(c, scale) for scale in scales]
    if motion_limits is not None:
        for frame_config, limits in zip(configs, motion_limits):
            frame_config.update({key: limits[key] for key in ('max_speed_px_frame', 'max_acceleration_px_frame2')})
    original = copy.deepcopy(rows)
    guarded = copy.deepcopy(rows)
    for i, row in enumerate(guarded):
        frame_config = configs[i]
        if finite(row.get('context_position')):
            allowed, info = allow_spot(original,i,np.asarray(row['context_position']),people,frame_config)
            row['penalty_spot_guard'] = dict(info, rejected=not allowed)
            if not allowed:
                row['context_position'] = None
                row['context_source'] = 'penalty_spot_rejected'
        for key in ('candidates','crowded_zoom_candidates','player_context_candidates','recovery_candidates','blank_recovery_candidates'):
            if row.get(key):
                row[key] = [p for p in row[key] if allow_spot(original,i,np.asarray(p[:2]),people,frame_config)[0]]
    if local_motion:
        from local_ball_motion import filter_local_motion
        local_limits = motion_limits or configs
        guarded = filter_local_motion(guarded, local_limits, scales,
            lambda i, point: allow_spot(original, i, point, people, configs[i])[0], people=people)
        for row in guarded:
            local = row['local_motion']
            if not local['accepted']:
                local['forward_prediction_position'] = row['context_position']
                # Future accepted image observations can correct a failed flight
                # forecast. Fill bridges these gaps in camera-compensated space.
                row['context_position'] = None
    result = fill(guarded,width,height)
    previous = velocity = None
    for i,row in enumerate(result):
        scale = scales[i]
        frame_config = configs[i]
        speed_limit = frame_config['max_speed_px_frame']
        acceleration_limit = frame_config['max_acceleration_px_frame2']
        # The viewer can always recover the exact pre-guard evidence.
        row['evidence_position'] = copy.deepcopy(original[i].get('context_position'))
        row['evidence_source'] = original[i].get('context_source','missing')
        row['penalty_spot_guard'] = guarded[i].get('penalty_spot_guard')
        local = row.get('local_motion')
        if local and not local['accepted']:
            row['inferred_position'] = True
            if local['state'] in ('predicted', 'stale'):
                row['inference']['basis'] = 'local_motion_filtered_bidirectional_gap'
                row['inference']['uncertainty_radius_px'] = max(
                    row['inference']['uncertainty_radius_px'], local['uncertainty_radius_px'])
                row['inference']['high_uncertainty'] |= local['missed_frames'] >= 12
        desired = np.asarray(row['context_position'],float)
        allowed, info = allow_spot(original,i,desired,people,frame_config)
        if row.get('inference'):
            row['inference']['uncertainty_radius_px'] *= scale
        adjusted_for_spot = False
        if row['inferred_position'] and not allowed:
            _,_,center = nearest_spot(row,desired)
            radial = desired-center
            if previous is not None:
                affine = row.get('camera_affine')
                preceding = np.asarray(affine)[:,:2]@previous + np.asarray(affine)[:,2] if affine is not None else previous
                radial = preceding-center
            if np.linalg.norm(radial) < 1e-6:
                radial = np.array([0.,-1.])
            desired = center + radial/np.linalg.norm(radial)*(frame_config['spot_radius_px']+2*scale)
            adjusted_for_spot = True
        affine = row.get('camera_affine')
        valid = bool(i and same_camera(result,i-1,i) and np.shape(affine)==(2,3)
                     and np.isfinite(affine).all()
                     and abs(np.linalg.det(np.asarray(affine)[:,:2]))>1e-8)
        scale_floor = False
        if valid and np.linalg.norm(np.asarray(affine)[:,:2]@velocity) > speed_limit+acceleration_limit:
            if motion_limits is None:
                valid = False
            else:
                # A contracting scale estimate must never cause an unbounded reset.
                # Keep the envelope conservative until bounded deceleration catches up.
                factor = np.linalg.norm(np.asarray(affine)[:,:2]@velocity) / (speed_limit+acceleration_limit)
                speed_limit *= factor
                acceleration_limit *= factor
                scale_floor = True
        if valid:
            linear = np.asarray(affine)[:,:2]
            origin = linear@previous + np.asarray(affine)[:,2]
            transported_velocity = linear@velocity
            desired_velocity = desired-origin
            if local_motion:
                # Brake early enough to settle on a stationary target rather
                # than orbiting it under a bounded acceleration controller.
                distance = float(np.linalg.norm(desired_velocity))
                desired_velocity = clip_norm(desired_velocity, np.sqrt(2*acceleration_limit*distance))
            velocity = bounded_velocity(desired_velocity,transported_velocity,
                                        speed_limit,acceleration_limit)
            displayed = origin+velocity
        else:
            displayed, velocity = desired, np.zeros(2)
        displacement = float(np.linalg.norm(displayed-np.asarray(row['context_position'])))
        limited = displacement > 1e-5
        if limited or adjusted_for_spot:
            row['inferred_position'] = True
            row['context_source'] = 'continuous_motion_limited'
            uncertainty = copy.deepcopy(row.get('inference') or {})
            uncertainty.update(basis='motion_constrained_estimate',
                               uncertainty_radius_px=max(uncertainty.get('uncertainty_radius_px',8.*scale), displacement+8.*scale),
                               high_uncertainty=displacement>=72*scale or uncertainty.get('high_uncertainty',False),
                               offscreen=not(0<=displayed[0]<width and 0<=displayed[1]<height),calibrated=False)
            row['inference'] = uncertainty
        row['motion_constraint'] = dict(adjusted=limited, displacement_px=displacement,
                                        penalty_avoidance=adjusted_for_spot,
                                        speed_px_frame=float(np.linalg.norm(velocity)),camera_reset=not valid,
                                        relative_zoom=scale, speed_limit_px_frame=speed_limit,
                                        acceleration_limit_px_frame2=acceleration_limit,
                                        spot_radius_px=frame_config['spot_radius_px'])
        if motion_limits is not None:
            limits = motion_limits[i]
            effective_scale = limits['pixels_per_meter'] * speed_limit / limits['max_speed_px_frame']
            row['motion_constraint'].update(
                pixels_per_meter=effective_scale, raw_pixels_per_meter=limits['raw_pixels_per_meter'],
                scale_source=limits['scale_source'], scale_transition_floor=scale_floor,
                max_speed_mps=limits['max_speed_mps'], max_acceleration_mps2=limits['max_acceleration_mps2'])
        row['context_position'] = displayed.tolist()
        previous = displayed
    return result


def run(args):
    cache,out = private_path(args.cache),private_path(args.out)
    if out.exists():raise ValueError('Use a new output directory')
    rows = read_rows(cache/'detections.jsonl')
    calibration = limits = None
    with np.load(args.features,allow_pickle=False) as data:
        if data['frame_ids'].tolist()!=[r['frame'] for r in rows]:raise ValueError('Frame alignment differs')
        meta=json.loads(str(data['metadata']))
        offsets,boxes=data['player_offsets'],data['player_boxes']
        people=[boxes[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
        if bool(args.motion_calibration) != bool(args.homography):
            raise ValueError('Supply both --motion-calibration and --homography')
        if args.motion_calibration:
            from calibrate_tactical_motion import frame_motion_limits, load_homographies
            calibration = json.loads(args.motion_calibration.read_text())
            provenance = meta.get('fusion_input_sha256', {})
            if (calibration['source_sha256'] != provenance.get('atd_raw')
                    or digest(args.homography) != provenance.get('homography')
                    or calibration['fps'] != meta['fps']):
                raise ValueError('Calibration, homography and feature source provenance must match')
            matrices = load_homographies(args.homography, data['frame_ids'])
            limits = frame_motion_limits(rows, data['atd_ball_m'], data['atd_players_m'],
                                         matrices, calibration, camera_zoom_scales(rows),
                                         meta['width'], meta['height'])
    result=constrain(rows,people,meta['width'],meta['height'],motion_limits=limits,local_motion=args.local_motion)
    summary=dict(frames=len(result),blank_frames=sum(not finite(r['context_position']) for r in result),
                 rejected_spot_positions=sum(bool((r.get('penalty_spot_guard') or {}).get('rejected')) for r in result),
                 spot_reasons=dict(Counter((r.get('penalty_spot_guard') or {}).get('reason') for r in result)),
                 motion_adjusted_frames=sum(r['motion_constraint']['adjusted'] for r in result),
                 penalty_avoiding_estimates=sum(r['motion_constraint']['penalty_avoidance'] for r in result),
                 inferred_frames=sum(r['inferred_position'] for r in result),
                 relative_zoom_range=[min(r['motion_constraint']['relative_zoom'] for r in result),max(r['motion_constraint']['relative_zoom'] for r in result)],
                 accuracy_measured=False)
    config=json.loads((cache/'config.json').read_text())
    config['continuous_trajectory']=dict(user_authorized_estimates=True,code_sha256=digest(__file__))
    config['constrained_trajectory']=dict(CONFIG,parent_sha256=digest(cache/'detections.jsonl'),
                                         limits_scale_with_zoom=True,zoom_reference='first frame of each camera segment',
                                         zoom_source='cumulative square root of camera-affine determinant')
    if calibration:
        config['constrained_trajectory'].update(motion_calibration=calibration,
            calibration_sha256=digest(args.motion_calibration), homography_sha256=digest(args.homography),
            scale_method='local homography maximum singular value, camera-aware smoothed scale')
        for key in ('speed_limit_px_frame', 'acceleration_limit_px_frame2', 'pixels_per_meter'):
            values = [r['motion_constraint'][key] for r in result]
            summary[key+'_range'] = [min(values), max(values)]
        summary['scale_transition_floor_frames'] = sum(r['motion_constraint']['scale_transition_floor'] for r in result)
    if args.local_motion:
        config['constrained_trajectory']['local_motion_gate'] = dict(
            horizon_frames=50, observation_noise_px_at_1x=3, innovation_squared_gate=9.21,
            process_noise='recent accepted-observation acceleration',
            bootstrap='three motion-consistent image observations',
            code_sha256=digest(Path(__file__).with_name('local_ball_motion.py')))
        summary['local_motion_states'] = dict(Counter(r['local_motion']['state'] for r in result))
    out.mkdir(parents=True)
    atomic(out/'config.json',config);atomic(out/'summary.json',summary)
    (out/'detections.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in result))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--motion-calibration',type=Path,help='Empirical metric envelopes from calibrate_tactical_motion.py')
    p.add_argument('--homography',type=Path,help='Same-camera tactical homography CSV for metric-to-pixel scaling')
    p.add_argument('--local-motion',action='store_true',help='Experimental local gate; regressed in paired tracking diagnostics (see EXPERIMENTS.md)')
    run(p.parse_args())

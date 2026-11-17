"""Point localization metrics, distinct from possession/spell metrics."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_points(predicted, target, radius):
    p, t = np.asarray(predicted, float), np.asarray(target, float)
    if (p.ndim != 2 or t.ndim != 2 or p.shape[1] != 2 or t.shape[1] != 2
            or not np.isfinite(p).all() or not np.isfinite(t).all()
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError('Invalid ball-point coordinates/radius')
    if not len(p) or not len(t):
        return []
    distances = np.linalg.norm(p[:, None]-t[None], axis=2)
    # Maximum valid cardinality first, minimum distance second. Invalid-pair
    # penalty exceeds the sum of all possible valid-pair distances.
    penalty = (min(len(p), len(t))+1)*(radius+1)
    a, b = linear_sum_assignment(np.where(distances <= radius, distances, penalty))
    return [(int(i), int(j), float(distances[i, j])) for i, j in zip(a, b) if distances[i, j] <= radius]


def score_image(predicted, target, radius):
    p = np.asarray(predicted, float)
    t = np.asarray(target, float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p[:, 2] < 0).any() or (p[:, 2] > 1).any():
        raise ValueError('Expected x/y/confidence predictions')
    matches = match_points(p[:, :2], t, radius)
    first = p[np.argmax(p[:, 2]), :2] if len(p) else None
    return dict(images=1, predictions=len(p), targets=len(t), true_positive=len(matches),
        false_positive=len(p)-len(matches), missed=len(t)-len(matches),
        annotated_ball_images=int(len(t) > 0), annotated_ball_free_images=int(len(t) == 0),
        ball_free_false_detections=len(p) if not len(t) else 0,
        top1_hit_images=int(first is not None and len(t) > 0 and np.linalg.norm(t-first, axis=1).min() <= radius),
        matched_distance_sum_px=float(sum(r[2] for r in matches)))


def aggregate(rows, uncertainty=True):
    if not rows:
        raise ValueError('No image scores')
    counts = {k: sum(row[k] for row in rows) for k in rows[0]}
    tp, pred, target, n = (counts[k] for k in ('true_positive', 'predictions', 'targets', 'images'))
    counts.update(point_precision=tp/pred if pred else None, point_recall=tp/target if target else None,
        point_f1=2*tp/(pred+target) if pred+target else None,
        false_detections_per_image=counts['false_positive']/n,
        top1_hit_fraction=counts['top1_hit_images']/counts['annotated_ball_images'] if counts['annotated_ball_images'] else None,
        mean_matched_distance_px=counts['matched_distance_sum_px']/tp if tp else None,
        ball_free_false_detections_per_image=counts['ball_free_false_detections']/counts['annotated_ball_free_images'] if counts['annotated_ball_free_images'] else None)
    if uncertainty:
        rng = np.random.default_rng(109)
        draws = np.array([[r['true_positive'], r['predictions'], r['targets']] for r in rows])
        totals = draws[rng.integers(len(rows), size=(2000, len(rows)))].sum(1)
        intervals = {}
        for name, den in (('point_precision', totals[:, 1]), ('point_recall', totals[:, 2])):
            values = totals[den > 0, 0]/den[den > 0]
            intervals[name] = np.percentile(values, [2.5, 97.5]).tolist() if len(values) else None
        counts['image_bootstrap_95_intervals'] = intervals
        counts['uncertainty_note'] = 'Images resampled within one match; temporal correlation may make these intervals optimistic.'
    return counts

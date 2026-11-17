"""Confidence/ground-validity handling before frozen possession features.

Image visibility is not ground-position validity. Baseline observations remain
unchanged; qualified CV observations may fill missing baseline positions only.
Unknown scores/heights are explicit, never silently treated as certainty.
"""
from dataclasses import dataclass
import numpy as np
from direct_weight_model import _blocks


@dataclass(frozen=True)
class QualityConfig:
    minimum_confidence: float = .25
    maximum_ground_sigma_m: float = 2.
    maximum_airborne_risk: float = .25
    maximum_anchor_gap_frames: int = 50
    anchor_tolerance_m: float = 2.

    def __post_init__(self):
        if (not 0<=self.minimum_confidence<=1 or not 0<=self.maximum_airborne_risk<=1 or
                not np.isfinite(self.maximum_ground_sigma_m) or self.maximum_ground_sigma_m<=0 or
                not np.isfinite(self.anchor_tolerance_m) or self.anchor_tolerance_m<=0 or
                type(self.maximum_anchor_gap_frames) is not int or self.maximum_anchor_gap_frames<1):
            raise ValueError('Invalid observation quality configuration')


def anchor_support(baseline,cv,blocks,config=QualityConfig()):
    """Short observed-endpoint consistency; not a ball-height measurement.

Interpolated anchor positions are used only as a diagnostic; never returned as
model observations. Halves/cuts are isolated and extrapolation is forbidden.
"""
    baseline=np.asarray(baseline,float);cv=np.asarray(cv,float)
    support=np.zeros(len(cv),bool);residual=np.full(len(cv),np.nan)
    for _,first,last in _blocks(blocks,len(cv)):
        indices=np.flatnonzero(np.isfinite(baseline[first:last+1]).all(1))+first
        for left,right in zip(indices[:-1],indices[1:]):
            if not 1<right-left<=config.maximum_anchor_gap_frames+1:continue
            rows=np.arange(left+1,right)
            fraction=(rows-left)/(right-left)
            expected=baseline[left]+fraction[:,None]*(baseline[right]-baseline[left])
            residual[rows]=np.linalg.norm(cv[rows]-expected,axis=1)
            support[rows]=np.isfinite(cv[rows]).all(1)&(residual[rows]<=config.anchor_tolerance_m)
    return support,residual


def select_observations(baseline,cv,confidence,ground_sigma_m,airborne_risk,blocks,config=QualityConfig()):
    """Return selected XY plus complete reason/visibility masks; no labels used."""
    baseline=np.asarray(baseline,float);cv=np.asarray(cv,float);n=len(cv)
    if baseline.shape!=(n,2) or cv.shape!=(n,2) or np.isinf(baseline).any() or np.isinf(cv).any():
        raise ValueError('Expected paired finite-or-missing XY observations')
    for points in (baseline,cv):
        if np.any(np.isfinite(points).any(1)!=np.isfinite(points).all(1)):
            raise ValueError('Coordinates must be jointly observed or missing')
    confidence,ground_sigma_m,airborne_risk=[np.asarray(x,float) for x in (confidence,ground_sigma_m,airborne_risk)]
    if any(x.shape!=(n,) for x in (confidence,ground_sigma_m,airborne_risk)):
        raise ValueError('Quality metadata must align one-to-one with frames')
    for score in (confidence,airborne_risk):
        if np.isinf(score).any() or np.any(np.isfinite(score)&((score<0)|(score>1))):
            raise ValueError('Scores must be in [0,1] or NaN for unknown')
    if np.any(ground_sigma_m<0):raise ValueError('Uncertainty cannot be negative')
    baseline_seen=np.isfinite(baseline).all(1);cv_seen=np.isfinite(cv).all(1)
    anchors,residual=anchor_support(baseline,cv,blocks,config)
    confident=np.isfinite(confidence)&(confidence>=config.minimum_confidence)
    localized=np.isfinite(ground_sigma_m)&(ground_sigma_m<=config.maximum_ground_sigma_m)
    airborne_known=np.isfinite(airborne_risk)
    ground_supported=np.where(airborne_known,airborne_risk<=config.maximum_airborne_risk,anchors)
    eligible=cv_seen&confident&localized&ground_supported
    fill=~baseline_seen&eligible
    selected=baseline.copy();selected[fill]=cv[fill]
    reason=np.full(n,'cv_missing',dtype='<U40')
    reason[cv_seen]='ground_validity_unknown'
    reason[cv_seen&airborne_known&~ground_supported]='airborne_ground_projection'
    reason[cv_seen&~localized]='ground_uncertainty'
    reason[cv_seen&~confident]='confidence_low_or_unknown'
    reason[fill]='cv_fill'
    reason[baseline_seen]='baseline_retained'
    return selected,dict(image_observed=cv_seen,baseline_observed=baseline_seen,
        ground_observed=np.isfinite(selected).all(1),cv_eligible=eligible,cv_used=fill,
        confidence_known=np.isfinite(confidence),airborne_known=airborne_known,
        short_anchor_supported=anchors,anchor_residual_m=residual,decision=reason)

"""Image-space search circle that always includes the supplied tactical prior.

The prior is automatic tracking, not ground truth. This plans where to run the
image detector; it does not relax the independent detection acceptance gate.
"""
import numpy as np


def plan_search_region(guide, forecast, width, height, radius=160.):
    def valid(point):
        if point is None:return None
        p=np.asarray(point,dtype=float)
        return p if p.shape==(2,) and np.isfinite(p).all() else None
    guide,forecast=valid(guide),valid(forecast)
    if guide is None and forecast is None:return None
    # Bound speculative motion lead, while never displacing a valid guide out
    # of the circle. Limits apply to the forecast lead, not the guide inclusion.
    if guide is not None and forecast is not None:
        delta=forecast-guide;distance=float(np.linalg.norm(delta))
        lead=delta*min(1.,radius/max(distance,1e-9))
        center=guide+lead/2
        r=float(radius+np.linalg.norm(lead)/2)
    else:
        center=guide if guide is not None else forecast;r=float(radius)
    x,y=center
    box=[max(0,int(np.floor(x-r))),max(0,int(np.floor(y-r))),
         min(width,int(np.ceil(x+r))),min(height,int(np.ceil(y+r)))]
    if box[0]>=box[2] or box[1]>=box[3]:return None
    return dict(center=center.tolist(),radius=r,crop=box,guide=guide.tolist() if guide is not None else None,
                guide_in_frame=bool(guide is not None and 0<=guide[0]<width and 0<=guide[1]<height),
                forecast=forecast.tolist() if forecast is not None else None,
                forecast_lead_capped=bool(guide is not None and forecast is not None and np.linalg.norm(forecast-guide)>radius))

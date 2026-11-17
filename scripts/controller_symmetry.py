"""Frozen submission inference: controller symmetry."""
import numpy as np


def swap_teams(features):
    x=np.asarray(features)
    if x.ndim!=2 or x.shape[1] not in (40,56,96) or not np.isfinite(x).all():
        raise ValueError('expected finite 40/56/96 observed feature matrix')
    result=x.copy()
    result[:,1]=x[:,2];result[:,2]=x[:,1];result[:,3]=-x[:,3]
    result[:,12]=x[:,13];result[:,13]=x[:,12]
    result[:,16:28]=x[:,28:40];result[:,28:40]=x[:,16:28]
    if x.shape[1]>=56:
        result[:,42:46]=-x[:,42:46]
        result[:,50:54]=-x[:,50:54]
    if x.shape[1]==96:
        result[:,56:76]=x[:,76:96];result[:,76:96]=x[:,56:76]
        result[:,[59,60,69,70,79,80,89,90]]*=-1
    return result


def average_predictions(original,transformed):
    a,b=np.asarray(original),np.asarray(transformed)
    if a.ndim!=2 or a.shape[1]!=3 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid H/A/D prediction shapes')
    if (a<0).any() or (b<0).any() or not np.allclose(a.sum(1),1.) or not np.allclose(b.sum(1),1.):
        raise ValueError('invalid H/A/D probabilities')
    return .5*(a+b[:,[1,0,2]])


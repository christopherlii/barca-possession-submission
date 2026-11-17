"""Fixed, image-only transforms for small-object preprocessing experiments."""
import numpy as np


def transform(image,mode):
    import cv2
    if mode=='fine_clahe':
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(16,16)).apply(lab[:,:,0])
        return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    if mode=='luminance_sharpen':
        yuv=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
        y=yuv[:,:,0];yuv[:,:,0]=cv2.addWeighted(y,1.6,cv2.GaussianBlur(y,(0,0),.8),-.6,0)
        return cv2.cvtColor(yuv,cv2.COLOR_YCrCb2BGR)
    if mode=='original':return image.copy()
    raise ValueError('Unknown preprocessing mode: '+mode)


def native_tiles(width,height,size=640,overlap=160):
    if size<32 or not 0<=overlap<size:raise ValueError('Invalid tile size/overlap')
    def starts(n):
        last=max(0,n-size)
        return sorted(set([*range(0,last+1,size-overlap),last]))
    return [(x,y,min(width,x+size),min(height,y+size)) for y in starts(height) for x in starts(width)]

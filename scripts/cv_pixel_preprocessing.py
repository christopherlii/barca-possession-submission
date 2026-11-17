"""Deterministic, stateless detector input transformations in OpenCV BGR order."""
import numpy as np


def detector_image(image, mode='none'):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected uint8 BGR image')
    if mode == 'none':
        return image
    if mode == 'zero-green':
        result = image.copy()
        result[:, :, 1] = 0
        return result
    raise ValueError('Unknown detector preprocessing')

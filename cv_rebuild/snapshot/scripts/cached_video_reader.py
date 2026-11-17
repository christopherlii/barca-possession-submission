"""Bounded, pixel-preserving frame reuse for offline recovery passes.

Avoid seeking back to a codec keyframe for every adjacent requested image.
Returned arrays are copies so downstream drawing/preprocessing cannot poison the
cache. Only frame-index seeking is supported; other properties are read-only.
"""
from collections import OrderedDict
import json


class CachedVideoCapture:
    def __init__(self, source, max_bytes=128 * 1024 * 1024, max_skip=16):
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(source))
        self.next_frame = 0
        self.decoder_next = 0
        self.max_bytes = max_bytes
        self.max_skip = max_skip
        self.cache = OrderedDict()
        self.bytes = 0
        self.stats = dict(requests=0, cache_hits=0, seeks=0, grabs=0, decoded=0)

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        if prop == self.cv2.CAP_PROP_POS_FRAMES:
            return self.next_frame
        return self.cap.get(prop)

    def set(self, prop, value):
        if prop != self.cv2.CAP_PROP_POS_FRAMES:
            raise ValueError('Cached reader supports frame-index seeks only')
        if value < 0 or int(value) != value:
            raise ValueError('Expected a nonnegative integer frame index')
        self.next_frame = int(value)
        return True

    def read(self):
        index = self.next_frame
        self.stats['requests'] += 1
        if index in self.cache:
            self.stats['cache_hits'] += 1
            self.cache.move_to_end(index)
            self.next_frame += 1
            return True, self.cache[index].copy()
        distance = index - self.decoder_next
        if 0 <= distance <= self.max_skip:
            for _ in range(distance):
                if not self.cap.grab():
                    return False, None
                self.decoder_next += 1
                self.stats['grabs'] += 1
        else:
            if not self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, index):
                return False, None
            self.decoder_next = index
            self.stats['seeks'] += 1
        ok, frame = self.cap.read()
        if not ok:
            return False, None
        self.decoder_next = self.next_frame = index + 1
        self.stats['decoded'] += 1
        if frame.nbytes <= self.max_bytes:
            while self.cache and self.bytes + frame.nbytes > self.max_bytes:
                _, old = self.cache.popitem(last=False)
                self.bytes -= old.nbytes
            self.cache[index] = frame.copy()
            self.bytes += frame.nbytes
        return True, frame

    def release(self):
        self.cap.release()
        self.cache.clear()
        self.bytes = 0
        print('Frame reader: ' + json.dumps(self.stats), flush=True)

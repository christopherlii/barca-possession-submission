"""Local signal-only audio cues and conservative, piecewise clock alignment.

No speech recognition, hosted services, match annotations or possession answers.
Whistle-like tones and energy surges are candidates, never confirmed events.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import wave

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    fft_size: int = 1024
    hop: int = 320
    whistle_low_hz: float = 2000.0
    whistle_high_hz: float = 4500.0
    whistle_band_fraction: float = 0.25
    whistle_peak_fraction: float = 0.12
    minimum_rms: float = 0.002
    minimum_tone_seconds: float = 0.08
    surge_ratio: float = 2.0
    surge_baseline_seconds: float = 5.0
    minimum_surge_seconds: float = 0.10
    alignment_block_seconds: float = 30.0
    alignment_min_correlation: float = 0.85
    alignment_peak_margin: float = 0.10
    alignment_exclusion_seconds: float = 1.0

    def __post_init__(self):
        values = asdict(self)
        if any(not np.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError('Audio parameters must be positive and finite')
        if self.hop > self.fft_size or self.whistle_high_hz >= self.sample_rate / 2:
            raise ValueError('Invalid audio sampling configuration')
        if self.whistle_low_hz >= self.whistle_high_hz:
            raise ValueError('Invalid whistle band')


def extract_audio(source, target, config=AudioConfig(), start_seconds=0.0, duration=None):
    """Decode locally to mono PCM. Errors never echo source data or ffmpeg logs."""
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-ss', str(start_seconds), '-i', str(source)]
    if duration is not None:
        command += ['-t', str(duration)]
    command += ['-map', '0:a:0', '-vn', '-ac', '1', '-ar', str(config.sample_rate),
                '-c:a', 'pcm_s16le', str(target)]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise ValueError('Audio extraction failed or input has no audio stream')


def read_pcm(path, expected_rate=16000):
    with wave.open(str(path), 'rb') as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, expected_rate):
            raise ValueError('Expected mono 16-bit PCM at configured sample rate')
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float32) / 32768


def features(samples, config=AudioConfig(), start_seconds=0.0):
    """Compact STFT features; only bounded blocks are materialized."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError('Expected finite mono audio samples')
    size, hop = config.fft_size, config.hop
    count = max(0, 1 + (len(samples) - size) // hop)
    result = {key: np.zeros(count, dtype=np.float64)
              for key in ('rms', 'band_fraction', 'peak_fraction')}
    result['times'] = start_seconds + (np.arange(count) * hop + size / 2) / config.sample_rate
    result['fingerprint'] = np.zeros((count, 4), dtype=np.float64)
    frequencies = np.fft.rfftfreq(size, 1 / config.sample_rate)
    whistle = (frequencies >= config.whistle_low_hz) & (frequencies <= config.whistle_high_hz)
    band_edges = (0, 500, 2000, 4500, config.sample_rate / 2 + 1)
    window = np.hanning(size)
    for first in range(0, count, 512):
        stop = min(first + 512, count)
        block = np.lib.stride_tricks.sliding_window_view(
            samples[first*hop:(stop-1)*hop+size], size)[::hop]
        power = np.abs(np.fft.rfft(block * window, axis=1)) ** 2
        total = power.sum(axis=1) + 1e-20
        band = power[:, whistle]
        result['rms'][first:stop] = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1))
        result['band_fraction'][first:stop] = band.sum(axis=1) / total
        result['peak_fraction'][first:stop] = band.max(axis=1) / (band.sum(axis=1) + 1e-20)
        for i, (low, high) in enumerate(zip(band_edges, band_edges[1:])):
            mask = (frequencies >= low) & (frequencies < high)
            result['fingerprint'][first:stop, i] = np.log1p(power[:, mask].sum(axis=1))
    return result


def mask_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def detect_cues(feature, config=AudioConfig()):
    if not len(feature['times']):
        return []
    rms = feature['rms']
    step = config.hop / config.sample_rate
    size = max(3, round(config.surge_baseline_seconds / step))
    baseline = pd.Series(rms).rolling(size, center=True, min_periods=1).median().to_numpy()
    masks = {
        'whistle_like_tone': ((rms >= config.minimum_rms) &
            (feature['band_fraction'] >= config.whistle_band_fraction) &
            (feature['peak_fraction'] >= config.whistle_peak_fraction), config.minimum_tone_seconds),
        'energy_surge': ((rms >= config.minimum_rms) &
            (rms >= config.surge_ratio * np.maximum(baseline, config.minimum_rms)),
            config.minimum_surge_seconds),
    }
    cues = []
    for kind, (mask, minimum) in masks.items():
        for lo, hi in mask_runs(mask):
            if (hi-lo) * step + 1e-9 < minimum:
                continue
            cues.append(dict(kind=kind, start_seconds=float(feature['times'][lo]-step/2),
                             end_seconds=float(feature['times'][hi-1]+step/2)))
    return sorted(cues, key=lambda row: (row['start_seconds'], row['kind']))


def normalized_correlation(reference, target):
    """Multiband, locally normalized cross-correlation at all valid target offsets."""
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n, m = len(target), len(reference)
    if m < 2 or n < m or reference.ndim != 2 or target.ndim != 2 or reference.shape[1] != target.shape[1]:
        return np.empty(0)
    centered = reference - reference.mean(axis=0)
    energy = np.sum(centered**2)
    if energy < 1e-12:
        return np.zeros(n-m+1)
    fft_size = 1 << (n+m-2).bit_length()
    convolution = np.fft.irfft(np.fft.rfft(target, n=fft_size, axis=0) *
        np.fft.rfft(centered[::-1], n=fft_size, axis=0), n=fft_size, axis=0)
    numerator = convolution[m-1:n].sum(axis=1)
    sums = np.vstack([np.zeros((1, target.shape[1])), np.cumsum(target, axis=0)])
    squares = np.vstack([np.zeros((1, target.shape[1])), np.cumsum(target**2, axis=0)])
    local_energy = (squares[m:]-squares[:-m]-(sums[m:]-sums[:-m])**2/m).sum(axis=1)
    return np.clip(numerator / np.sqrt(energy*np.maximum(local_energy, 1e-12)), -1, 1)


def align_blocks(reference, broadcast, config=AudioConfig()):
    """Only matched blocks get a mapping. No extrapolation across cuts or replays.

    A high match score AND separation from the second-best distant peak are required.
    Correlations are diagnostics, not calibrated probabilities.
    """
    step = config.hop / config.sample_rate
    size = max(2, round(config.alignment_block_seconds / step))
    exclusion = max(1, round(config.alignment_exclusion_seconds / step))
    blocks = []
    for lo in range(0, len(reference['times']), size):
        hi = min(lo + size, len(reference['times']))
        if hi-lo < max(2, round(5 / step)):
            continue
        scores = normalized_correlation(reference['fingerprint'][lo:hi], broadcast['fingerprint'])
        if not len(scores):
            continue
        peak = int(np.argmax(scores))
        alternative = scores.copy()
        alternative[max(0, peak-exclusion):peak+exclusion+1] = -1
        second = float(np.max(alternative))
        best = float(scores[peak])
        if best < config.alignment_min_correlation or best-second < config.alignment_peak_margin:
            continue
        reference_time = float(reference['times'][lo])
        broadcast_time = float(broadcast['times'][peak])
        blocks.append(dict(broadcast_start_seconds=broadcast_time-step/2,
            broadcast_end_seconds=float(broadcast['times'][peak+hi-lo-1]+step/2),
            tactical_offset_seconds=reference_time-broadcast_time,
            correlation=best, second_peak=second))
    return blocks


def map_broadcast_cues(cues, blocks):
    mapped = []
    for cue in cues:
        # Refuse cues straddling an unmatched section or camera edit.
        hits = [b for b in blocks if b['broadcast_start_seconds'] <= cue['start_seconds']
                and cue['end_seconds'] <= b['broadcast_end_seconds']]
        if len(hits) != 1:
            continue
        offset = hits[0]['tactical_offset_seconds']
        mapped.append(dict(kind=cue['kind'], start_seconds=cue['start_seconds']+offset,
                           end_seconds=cue['end_seconds']+offset))
    return mapped


def audio_summary(feature, config=AudioConfig()):
    rms = feature['rms']
    return dict(feature_frames=len(rms), audible_fraction=float(np.mean(rms >= config.minimum_rms))
                if len(rms) else 0.0, rms_p95=float(np.quantile(rms, .95)) if len(rms) else 0.0)

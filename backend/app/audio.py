from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, wiener


TARGET_SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 30
N_MELS = 64
N_MFCC = 13
WINDOW_SECONDS = 3
HOP_SECONDS = 2
FEATURE_FRAMES = 300
VAD_FRAME_SECONDS = 0.02
VAD_HOP_SECONDS = 0.01
VAD_MIN_ACTIVE_SECONDS = 0.1
VAD_PADDING_SECONDS = 0.08


class AudioInputError(ValueError):
    """Raised when uploaded audio cannot safely be analysed."""


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    original_sample_rate: int
    original_channels: int
    original_duration: float


def denoise_wiener(samples: np.ndarray) -> np.ndarray:
    if samples.size < 3:
        return samples.copy()
    filtered = wiener(samples, mysize=15)
    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(filtered)))
    if peak <= 1e-8:
        raise AudioInputError("The Wiener denoiser produced no usable audio signal.")
    return np.ascontiguousarray(filtered / peak, dtype=np.float32)


def decode_wav(contents: bytes) -> AudioData:
    if not contents:
        raise AudioInputError("The selected audio file is empty.")

    try:
        file_info = sf.info(BytesIO(contents))
        if file_info.format != "WAV":
            raise AudioInputError("Only WAV audio files are supported.")
        audio, sample_rate = sf.read(BytesIO(contents), dtype="float32", always_2d=True)
    except AudioInputError:
        raise
    except (RuntimeError, sf.SoundFileError, ValueError) as exc:
        raise AudioInputError("This file is not a readable PCM WAV audio file.") from exc

    if audio.size == 0 or sample_rate <= 0:
        raise AudioInputError("The audio file contains no readable samples.")
    if audio.shape[1] > 2:
        raise AudioInputError("Audio with more than two channels is not supported.")
    if sample_rate < 8_000 or sample_rate > 96_000:
        raise AudioInputError("Sample rates from 8 kHz to 96 kHz are supported.")
    if not np.isfinite(audio).all():
        raise AudioInputError("The audio contains invalid numerical values.")

    original_duration = audio.shape[0] / sample_rate
    if original_duration > MAX_DURATION_SECONDS:
        raise AudioInputError(f"Audio must be {MAX_DURATION_SECONDS} seconds or shorter.")

    mono = audio.mean(axis=1, dtype=np.float32)
    if mono.size < int(sample_rate * 0.05):
        raise AudioInputError("Audio must be at least 0.05 seconds long.")
    if float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) < 1e-5:
        raise AudioInputError("No audible signal was detected; the recording may be silent.")

    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)

    peak = float(np.max(np.abs(mono)))
    if peak > 0:
        mono = mono / peak

    return AudioData(
        samples=np.ascontiguousarray(mono, dtype=np.float32),
        sample_rate=TARGET_SAMPLE_RATE,
        original_sample_rate=int(sample_rate),
        original_channels=int(audio.shape[1]),
        original_duration=float(original_duration),
    )


def _model_features(samples: np.ndarray) -> np.ndarray:
    minimum_samples = int(TARGET_SAMPLE_RATE * WINDOW_SECONDS)
    if samples.size < minimum_samples:
        samples = np.pad(samples, (0, minimum_samples - samples.size))
    else:
        samples = samples[:minimum_samples]

    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=TARGET_SAMPLE_RATE,
        n_fft=400,
        hop_length=160,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = np.clip(log_mel, -80.0, 0.0).astype(np.float32)
    if log_mel.shape[1] < FEATURE_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, FEATURE_FRAMES - log_mel.shape[1])))
    return log_mel[:, :FEATURE_FRAMES]


def detect_voice_activity(samples: np.ndarray) -> dict[str, object]:
    if samples.size == 0 or not np.isfinite(samples).all():
        raise AudioInputError("Voice activity cannot be estimated from invalid audio.")
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    frame_size = max(1, int(round(VAD_FRAME_SECONDS * TARGET_SAMPLE_RATE)))
    hop_size = max(1, int(round(VAD_HOP_SECONDS * TARGET_SAMPLE_RATE)))
    if samples.size < frame_size:
        framed = np.pad(samples, (0, frame_size - samples.size))[None, :]
    else:
        frame_count = 1 + (samples.size - frame_size) // hop_size
        shape = (frame_count, frame_size)
        strides = (samples.strides[0] * hop_size, samples.strides[0])
        frames = np.lib.stride_tricks.as_strided(
            samples, shape=shape, strides=strides, writeable=False
        )
        framed = frames
    frame_rms = np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=1))
    peak_rms = float(np.max(frame_rms, initial=0.0))
    threshold = peak_rms * 0.02
    active = frame_rms >= threshold if threshold > 0 else np.zeros_like(frame_rms, dtype=bool)
    padding_frames = int(np.ceil(VAD_PADDING_SECONDS / VAD_HOP_SECONDS))
    if active.any() and padding_frames:
        padded_active = active.copy()
        for shift in range(1, padding_frames + 1):
            padded_active[shift:] |= active[:-shift]
            padded_active[:-shift] |= active[shift:]
        active = padded_active

    padded = np.pad(active.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    minimum_frames = int(np.ceil(VAD_MIN_ACTIVE_SECONDS / VAD_HOP_SECONDS))
    duration = samples.size / TARGET_SAMPLE_RATE
    intervals = [
        [
            round(float(start * VAD_HOP_SECONDS), 3),
            round(min(float(end * VAD_HOP_SECONDS), duration), 3),
        ]
        for start, end in zip(starts, ends, strict=True)
        if end - start >= minimum_frames
    ]
    active_duration = sum(end - start for start, end in intervals)
    return {
        "method": "adaptive frame-energy estimate",
        "frame_seconds": VAD_FRAME_SECONDS,
        "hop_seconds": VAD_HOP_SECONDS,
        "relative_rms_threshold": 0.02,
        "active_intervals_seconds": intervals,
        "estimated_active_duration_seconds": round(active_duration, 3),
        "estimated_active_fraction": min(1.0, active_duration / duration) if duration else 0.0,
        "voice_detected": bool(intervals),
        "limitation": "Energy-based activity estimate; background noise may be marked active and quiet speech may be missed.",
    }


def extract_features(samples: np.ndarray) -> dict[str, object]:
    if samples.size == 0 or not np.isfinite(samples).all():
        raise AudioInputError("Audio features cannot be calculated for invalid samples.")

    n_fft = 400
    hop_length = 160
    mel = librosa.feature.melspectrogram(
        y=samples,
        sr=TARGET_SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = np.clip(librosa.power_to_db(mel, ref=np.max), -80.0, 0.0)
    mfcc = librosa.feature.mfcc(
        y=samples,
        sr=TARGET_SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    rms = librosa.feature.rms(y=samples, frame_length=n_fft, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(
        y=samples, sr=TARGET_SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y=samples, frame_length=n_fft, hop_length=hop_length
    )[0]
    try:
        pitches = librosa.yin(
            samples,
            fmin=70,
            fmax=400,
            sr=TARGET_SAMPLE_RATE,
            frame_length=1024,
            hop_length=hop_length,
        )
        pitch_hz = float(np.median(pitches)) if pitches.size else None
    except (ValueError, FloatingPointError):
        pitch_hz = None

    waveform = samples[:: max(1, int(np.ceil(samples.size / 1200)))]
    duration = samples.size / TARGET_SAMPLE_RATE
    envelope = librosa.feature.rms(y=samples, frame_length=n_fft, hop_length=hop_length)[0]

    def finite_mean(values: np.ndarray) -> float | None:
        valid = values[np.isfinite(values)]
        return float(np.mean(valid)) if valid.size else None

    output = {
        "voice_activity": detect_voice_activity(samples),
        "waveform": [float(value) for value in waveform],
        "log_mel": np.round(log_mel, 2).tolist(),
        "mfcc": np.round(mfcc, 2).tolist(),
        "frame_hop_seconds": hop_length / TARGET_SAMPLE_RATE,
        "acoustic": {
            "pitch_median_hz": pitch_hz,
            "rms_mean": finite_mean(rms),
            "spectral_centroid_mean_hz": finite_mean(centroid),
            "zero_crossing_rate_mean": finite_mean(zcr),
            "speech_rate_proxy_peaks_per_second": float(
                np.count_nonzero(
                    (envelope[1:-1] > envelope[:-2])
                    & (envelope[1:-1] >= envelope[2:])
                    & (envelope[1:-1] > np.mean(envelope))
                )
                / duration
            )
            if envelope.size > 2 and duration > 0
            else None,
        },
    }
    return output


def segment_audio(samples: np.ndarray) -> list[tuple[float, float, np.ndarray]]:
    window_size = int(WINDOW_SECONDS * TARGET_SAMPLE_RATE)
    hop_size = int(HOP_SECONDS * TARGET_SAMPLE_RATE)
    segments: list[tuple[float, float, np.ndarray]] = []
    for start_sample in range(0, samples.size, hop_size):
        end_sample = min(start_sample + window_size, samples.size)
        clip = samples[start_sample:end_sample]
        if clip.size == 0:
            continue
        padded = np.pad(clip, (0, max(0, window_size - clip.size)))
        segments.append(
            (
                start_sample / TARGET_SAMPLE_RATE,
                end_sample / TARGET_SAMPLE_RATE,
                _model_features(padded),
            )
        )
        if end_sample == samples.size:
            break
    return segments

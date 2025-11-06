"""Audio utilities for TTS preprocessing."""
import numpy as np


def ensure_minimum_duration(audio: np.ndarray, sr: int, min_duration: float = 3.0) -> np.ndarray:
    """
    Ensure audio meets minimum duration requirement by repeating if necessary.

    Args:
        audio: Audio waveform as numpy array
        sr: Sample rate in Hz
        min_duration: Minimum duration in seconds (default: 3.0)

    Returns:
        Audio array that meets minimum duration requirement
    """
    current_duration = len(audio) / sr

    if current_duration >= min_duration:
        return audio

    min_samples = int(sr * min_duration)
    num_repeats = int(np.ceil(min_samples / len(audio)))

    repeated_audio = np.tile(audio, num_repeats)
    return repeated_audio[:min_samples]
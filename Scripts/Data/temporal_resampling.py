from __future__ import annotations

import numpy as np


def _smooth_1d(values: np.ndarray, window: int) -> np.ndarray:
    if values.size <= 1:
        return values
    kernel_size = max(1, int(window))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1:
        return values
    kernel = np.ones((kernel_size,), dtype=np.float64) / float(kernel_size)
    padded = np.pad(values.astype(np.float64), (kernel_size // 2, kernel_size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def build_temporal_source_positions(
    num_source_steps: int,
    speed_scale: float,
    warp_strength: float,
    warp_smooth_window: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build endpoint-aligned source positions for fixed-rate temporal resampling.

    The number of output intervals is approximately source_intervals / speed.
    Local warp changes the relative interval sizes, while normalization keeps the
    first and last positions exactly aligned with the source trajectory.
    """
    source_steps = max(1, int(num_source_steps))
    if source_steps <= 1:
        return np.zeros((1,), dtype=np.float64)

    speed = float(speed_scale)
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError("speed_scale must be a finite value greater than zero.")

    source_span = float(source_steps - 1)
    output_intervals = max(1, int(np.floor(source_span / speed + 0.5)))
    strength = float(max(0.0, warp_strength))

    if strength <= 1.0e-6:
        return np.linspace(0.0, source_span, output_intervals + 1, dtype=np.float64)

    noise = rng.normal(0.0, 1.0, size=(output_intervals,))
    noise = _smooth_1d(noise, window=max(1, int(warp_smooth_window)))
    max_abs = float(np.max(np.abs(noise))) if noise.size > 0 else 0.0
    if max_abs > 1.0e-12:
        noise = noise / max_abs

    interval_weights = np.clip(1.0 + strength * noise, 0.2, 3.0)
    weight_sum = float(np.sum(interval_weights))
    if not np.isfinite(weight_sum) or weight_sum <= 1.0e-12:
        interval_weights = np.ones((output_intervals,), dtype=np.float64)
        weight_sum = float(output_intervals)

    increments = interval_weights * (source_span / weight_sum)
    source_positions = np.empty((output_intervals + 1,), dtype=np.float64)
    source_positions[0] = 0.0
    source_positions[1:] = np.cumsum(increments, dtype=np.float64)
    source_positions[-1] = source_span
    return source_positions

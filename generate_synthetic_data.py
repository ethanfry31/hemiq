import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_transport_data(
    num_samples=120,
    dt_seconds=60,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a synthetic transport log for blood transport testing.

    The output contains columns expected by `src/main.py`:
    - dt: time step in seconds
    - t_surface: surface temperature in degrees Celsius
    - ax, ay, az: acceleration components in m/s^2
    """
    rng = np.random.default_rng(seed)

    # Time base
    dt = float(dt_seconds)
    times = np.arange(num_samples) * dt

    # Temperature profile: mostly refrigerated, with a few warm-up events.
    base_temp = 4.0
    temp_profile = np.full(num_samples, base_temp, dtype=float)

    warm_sections = [
        (20, 30, 15.0),
        (60, 70, 22.0),
        (90, 100, 18.0),
    ]
    for start, end, peak in warm_sections:
        duration = end - start
        ramp_up = np.linspace(base_temp, peak, duration // 2, endpoint=False)
        ramp_down = np.linspace(peak, base_temp, duration - len(ramp_up), endpoint=True)
        temp_profile[start:end] = np.concatenate((ramp_up, ramp_down))

    # Add gentle ambient drift and noise
    temp_profile += rng.normal(loc=0.0, scale=0.2, size=num_samples)
    temp_profile = np.clip(temp_profile, 0.0, 25.0)

    # Vibrational profile: low-level transport vibration with occasional bumps
    def make_axis_noise(base_std=0.3, bump_std=1.2):
        axis = rng.normal(0.0, base_std, size=num_samples)
        for bump_center in [15, 45, 75, 110]:
            window = np.arange(num_samples)
            bump = np.exp(-0.5 * ((window - bump_center) / 4.0) ** 2)
            axis += bump * rng.normal(0.0, bump_std)
        return axis

    ax = make_axis_noise()
    ay = make_axis_noise()
    az = make_axis_noise()

    # Smooth the acceleration signals slightly
    def smooth(signal, window=5):
        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode="same")

    ax = smooth(ax)
    ay = smooth(ay)
    az = smooth(az)

    # Ensure there is a realistic baseline offset from gravity cancellation
    ax += rng.normal(0.0, 0.06, size=num_samples)
    ay += rng.normal(0.0, 0.06, size=num_samples)
    az += rng.normal(0.0, 0.06, size=num_samples)

    return pd.DataFrame(
        {
            "dt": np.full(num_samples, dt),
            "t_surface": temp_profile.round(2),
            "ax": ax.round(3),
            "ay": ay.round(3),
            "az": az.round(3),
        }
    )


if __name__ == "__main__":
    output_path = Path("data") / "synthetic_transport_log.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_transport_data(num_samples=120, dt_seconds=60, seed=123)
    df.to_csv(output_path, index=False)
    print(f"Synthetic test data written to: {output_path.resolve()}")

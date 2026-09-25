"""Prepare the clock samples from the chosen 44.1 kHz mono TikTok WAV.

Run with: uv run python src/browser/webclient/tools/prepare_clock_audio.py SOURCE.wav
We retain the source pitch and suppress the music with a smooth spectral mask.
We cannot recover a separate clock stem from a mono mix without some loss.
"""
from argparse import ArgumentParser
from pathlib import Path
import wave

import numpy as np
from scipy import ndimage, signal


def read(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path)) as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 44100):
            raise ValueError("Use the 44.1 kHz, mono, 16-bit source WAV.")
        return source.getframerate(), np.frombuffer(source.readframes(source.getnframes()), "<i2") / 32768


def cut(source: np.ndarray, rate: int, start: float, end: float,
        noise_start: float, noise_end: float, peak: float, fade: float,
        decay_only: bool = False) -> np.ndarray:
    # We retain the attack, then suppress frequencies present in the music bed.
    segment = source[int(start * rate):int(end * rate)]
    noise = source[int(noise_start * rate):int(noise_end * rate)]
    options = dict(fs=rate, nperseg=1024, noverlap=896)
    _, times, spectrum = signal.stft(segment, **options)
    _, _, bed = signal.stft(noise, **options)
    power = np.abs(spectrum) ** 2
    floor = np.quantile(np.abs(bed) ** 2, 0.8, axis=1, keepdims=True)
    gain = np.sqrt(np.maximum(0, 1 - 2.5 * floor / (power + 1e-12)))
    gain = ndimage.gaussian_filter(gain, sigma=(1.0, 1.0))
    if decay_only:
        # After the gong attack, we retain decaying frequencies and reject
        # new notes. We smooth the energy to retain the gong's beating modes.
        clean_power = ndimage.gaussian_filter(power * gain**2, sigma=(1.0, 4.0))
        first = int(np.searchsorted(times, 0.45))
        ceiling = clean_power[:, first].copy()
        decay = np.exp(-2 * (times[1] - times[0]) / 0.8)
        for frame in range(first, len(times)):
            ceiling = np.minimum(ceiling * decay, clean_power[:, frame])
            gain[:, frame] *= np.sqrt(np.minimum(1, ceiling / (power[:, frame] + 1e-12)))
    _, clean = signal.istft(spectrum * gain, **options)
    clean = clean[:len(segment)]
    clean = signal.sosfilt(signal.butter(2, [70, 5500], btype="bandpass", fs=rate, output="sos"), clean)
    attack = int(0.002 * rate)
    tail = int(fade * rate)
    clean[:attack] *= np.linspace(0, 1, attack)
    clean[-tail:] *= np.cos(np.linspace(0, np.pi / 2, tail)) ** 2
    clean *= peak / max(float(np.max(np.abs(clean))), 1e-12)
    return clean


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    rate, source = read(args.source)
    if len(source) < int(24.28 * rate):
        raise ValueError("The source must include the gong at 22.84 seconds.")
    output = Path(__file__).resolve().parents[1] / "public" / "audio"
    for name, settings in {
        "tick": (0.180, 0.360, 0.50, 0.95, 0.48, 0.08, False),
        "gong": (22.840, 24.280, 22.30, 22.83, 0.70, 0.35, True),
    }.items():
        samples = cut(source, rate, *settings)
        with wave.open(str(output / f"{name}.wav"), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(rate)
            target.writeframes(np.round(samples * 32767).astype("<i2").tobytes())
        print(f"{name}: {len(samples) / rate:.3f}s, peak {np.max(np.abs(samples)):.2f}")


if __name__ == "__main__":
    main()

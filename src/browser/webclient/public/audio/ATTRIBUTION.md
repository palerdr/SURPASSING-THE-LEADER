# Audio attribution

At your request, we cut `tick.wav` and `gong.wav` from the TikTok video
https://www.tiktok.com/@2kys3/video/7500673204722208007 by @2kys3.
The recording is an Usogui leap-second edit, credited as original sound.

We use the first tick at 0.180–0.360 seconds, which includes its initial
attack and low resonance. We fade its last 80 ms to exclude the faint
music-bearing tail. We repeat this tick at its source pitch.
We no longer play `tock.wav`, the earlier pitch-shifted copy.
We cut the gong at 22.840–24.280 seconds and fade its last 350 ms. After
450 ms, we constrain each frequency to a decaying energy envelope to
suppress new musical notes during the ring. We leave the first 400 ms
unchanged and let the tail finish after a timeout.

We estimate the tick's music bed from 0.50–0.95 seconds and the gong's bed
from 22.30–22.83 seconds. We apply a smooth spectral mask and a 70–5500 Hz
bandpass, with a 2 ms attack fade. We cap sample peaks at 0.48 for the tick
and 0.70 for the gong to leave headroom. The source mixes music and clock
into one mono channel. This treatment suppresses music; it cannot recover
an exact isolated clock stem. Listening validation remains open.

Use the existing NumPy and SciPy environment to reproduce the cuts:

```bash
uv run python src/browser/webclient/tools/prepare_clock_audio.py /path/to/ref.wav
```

The input is a mono, 16-bit, 44.1 kHz WAV, 41.1893 seconds long. Its SHA-256 is
`50cf1583e16f58bf3df36fd557bbf11f243bae0b6711331c0b0625836bd8eb8a`.
The script does not download the source.

The rights to this audio remain uncleared for redistribution. These cuts
serve your local play; clear the rights before a public release.

The earlier tick, “Tower clock from the Holy Trinity Church in Gdańsk” by
Work With Sounds / Museum of Municipal Engineering (Wikimedia Commons,
CC BY 4.0), is no longer in use.

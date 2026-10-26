"""Audio alert module for audible industrial alarm triggers."""

from __future__ import annotations
import io
import math
import wave
import struct


def generate_alarm_wav_bytes(frequency: int = 880, duration_seconds: float = 0.5, sample_rate: int = 22050) -> bytes:
    """Generates an in-memory WAV audio beep for web dashboard and alerting systems."""
    num_samples = int(duration_seconds * sample_rate)
    buf = io.BytesIO()

    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)

        for i in range(num_samples):
            # Two-tone pulsating siren
            t = float(i) / sample_rate
            mod_freq = frequency if int(t * 8) % 2 == 0 else int(frequency * 1.25)
            val = int(32767.0 * 0.45 * math.sin(2.0 * math.pi * mod_freq * t))
            data = struct.pack("<h", max(-32768, min(32767, val)))
            wav.writeframesraw(data)

    return buf.getvalue()

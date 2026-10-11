import librosa
import numpy as np


def analyze_speech(path):
    # Load audio
    y, sr = librosa.load(path, sr=None)

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    avg_energy = float(np.mean(rms))

    # Pitch / F0
    f0, _voiced_flag, _voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )

    pitch = f0[~np.isnan(f0)]

    avg_pitch = float(np.mean(pitch)) if len(pitch) > 0 else 0.0

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    mfcc_mean = np.mean(mfcc, axis=1)

    # FFT
    fft = np.abs(np.fft.rfft(y))
    fft_mean = float(np.mean(fft))

    # Detect silence / pauses
    intervals = librosa.effects.split(y, top_db=30)

    speech_time = 0

    for start, end in intervals:
        speech_time += (end - start) / sr

    pause_time = max(0, duration - speech_time)

    # Estimate speech rate
    # This is a simple first version.
    estimated_words = max(1, int(speech_time * 2.5))
    speech_rate = estimated_words / (speech_time / 60)

    result = {
        "duration": round(duration, 2),
        "speech_time": round(speech_time, 2),
        "pause_time": round(pause_time, 2),
        "speech_rate_wpm": round(speech_rate, 2),
        "average_pitch_hz": round(avg_pitch, 2),
        "average_energy": round(avg_energy, 4),
        "fft_mean": round(fft_mean, 4),
        "mfcc": [round(float(x), 4) for x in mfcc_mean],
    }

    return result


if __name__ == "__main__":
    print("Speech Analysis Module")
    print("Ready to analyze an audio file.")

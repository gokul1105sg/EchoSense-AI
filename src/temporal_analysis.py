import librosa
import numpy as np


def analyze_temporal(file):
    y, sr = librosa.load(file, sr=16000)

    hop = 512

    # Energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]

    # Pitch
    pitch, _, _ = librosa.pyin(
        y,
        fmin=70,
        fmax=400,
        sr=sr,
        hop_length=hop
    )

    # Time for each frame
    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop
    )

    # Replace missing pitch values
    pitch_clean = np.nan_to_num(pitch, nan=0.0)

    # Normalize energy
    energy_threshold = np.percentile(rms, 20)

    result = []

    for i in range(len(times)):
        result.append({
            "time": round(float(times[i]), 2),
            "energy": round(float(rms[i]), 5),
            "pitch": round(float(pitch_clean[i]), 2),
            "speaking": bool(rms[i] > energy_threshold)
        })

    return result


if __name__ == "__main__":
    data = analyze_temporal("dataset/ideal/speech1.wav")

    print("Number of frames:", len(data))

    print("\nFirst 10 frames:")

    for row in data[:10]:
        print(row)
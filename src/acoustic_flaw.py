import librosa
import numpy as np


def get_acoustic_features(file, start, end):
    y, sr = librosa.load(file, sr=16000)

    # Convert seconds to samples
    s = int(start * sr)
    e = int(end * sr)

    segment = y[s:e]

    if len(segment) == 0:
        return {"pitch": 0, "energy": 0, "pitch_variation": 0}

    # -------------------------
    # ENERGY
    # -------------------------
    rms = librosa.feature.rms(y=segment, frame_length=1024, hop_length=256)[0]

    energy = float(np.mean(rms))

    # -------------------------
    # PITCH / F0
    # -------------------------
    pitch, _, _ = librosa.pyin(segment, fmin=70, fmax=400, sr=sr, hop_length=256)

    valid_pitch = pitch[~np.isnan(pitch)]

    if len(valid_pitch) > 0:
        avg_pitch = float(np.mean(valid_pitch))
        pitch_variation = float(np.std(valid_pitch))
    else:
        avg_pitch = 0
        pitch_variation = 0

    return {
        "pitch": round(avg_pitch, 2),
        "energy": round(energy, 5),
        "pitch_variation": round(pitch_variation, 2),
    }


if __name__ == "__main__":
    ideal_file = "dataset/ideal/speech1.wav"
    flawed_file = "dataset/flawed/flawed1.wav"

    start = 1.04
    end = 4.40

    ideal = get_acoustic_features(ideal_file, start, end)

    flawed = get_acoustic_features(flawed_file, start, end)

    print("\nACOUSTIC ANALYSIS")
    print("=================")

    print("\nIDEAL")
    print("Pitch:", ideal["pitch"], "Hz")
    print("Energy:", ideal["energy"])
    print("Pitch variation:", ideal["pitch_variation"])

    print("\nFLAWED")
    print("Pitch:", flawed["pitch"], "Hz")
    print("Energy:", flawed["energy"])
    print("Pitch variation:", flawed["pitch_variation"])

    print("\nDIFFERENCE")

    pitch_diff = abs(ideal["pitch"] - flawed["pitch"])

    energy_diff = abs(ideal["energy"] - flawed["energy"])

    print("Pitch difference:", round(pitch_diff, 2), "Hz")
    print("Energy difference:", round(energy_diff, 5))

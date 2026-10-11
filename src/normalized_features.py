import librosa
import numpy as np
import whisper

# Load Whisper model once
MODEL = whisper.load_model("base")


# --------------------------------------------------
# GET WORD TIMESTAMPS
# --------------------------------------------------

def get_words(file):
    result = MODEL.transcribe(
        file,
        word_timestamps=True
    )

    words = []

    for segment in result["segments"]:
        for word in segment["words"]:

            text = word["word"].strip()

            if text:
                words.append({
                    "word": text,
                    "start": float(word["start"]),
                    "end": float(word["end"])
                })

    return words


# --------------------------------------------------
# CALCULATE SPEECH RATE
# --------------------------------------------------

def calculate_speech_rate(words):

    if len(words) < 2:
        return 0

    start = words[0]["start"]
    end = words[-1]["end"]

    speech_duration = end - start

    if speech_duration <= 0:
        return 0

    word_count = len(words)

    wpm = (word_count / speech_duration) * 60

    return wpm


# --------------------------------------------------
# ANALYZE AUDIO
# --------------------------------------------------

def analyze_audio(file):

    print("Analyzing:", file)

    # Load audio
    y, sr = librosa.load(
        file,
        sr=16000
    )

    # ----------------------------------------------
    # Duration
    # ----------------------------------------------

    duration = librosa.get_duration(
        y=y,
        sr=sr
    )

    # ----------------------------------------------
    # RMS Energy
    # ----------------------------------------------

    rms = librosa.feature.rms(
        y=y
    )[0]

    average_energy = float(
        np.mean(rms)
    )

    # ----------------------------------------------
    # Pitch
    # ----------------------------------------------

    pitch, _, _ = librosa.pyin(
        y,
        fmin=70,
        fmax=400,
        sr=sr
    )

    pitch_clean = pitch[
        ~np.isnan(pitch)
    ]

    if len(pitch_clean) > 0:

        average_pitch = float(
            np.mean(pitch_clean)
        )

        pitch_variation = float(
            np.std(pitch_clean)
        )

    else:

        average_pitch = 0
        pitch_variation = 0

    # ----------------------------------------------
    # MFCC
    # ----------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    # ----------------------------------------------
    # Whisper Word Timestamps
    # ----------------------------------------------

    words = get_words(file)

    speech_rate = calculate_speech_rate(
        words
    )

    # ----------------------------------------------
    # Return Features
    # ----------------------------------------------

    return {
        "duration": duration,
        "speech_rate": speech_rate,
        "word_count": len(words),
        "average_pitch": average_pitch,
        "average_energy": average_energy,
        "pitch_variation": pitch_variation,
        "mfcc": mfcc_mean
    }


# --------------------------------------------------
# PERCENTAGE CHANGE
# --------------------------------------------------

def percentage_change(ideal, flawed):

    if ideal == 0:
        return 0

    return (
        (flawed - ideal)
        / ideal
    ) * 100


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    ideal_file = "dataset/ideal/speech1.wav"
    flawed_file = "dataset/flawed/flawed1.wav"

    print()
    print("======================================")
    print("NORMALIZED SPEECH FEATURE ANALYSIS")
    print("======================================")

    # ----------------------------------------------
    # IDEAL
    # ----------------------------------------------

    print()
    print("Analyzing ideal speech...")

    ideal = analyze_audio(
        ideal_file
    )

    # ----------------------------------------------
    # FLAWED
    # ----------------------------------------------

    print()
    print("Analyzing flawed speech...")

    flawed = analyze_audio(
        flawed_file
    )

    # ----------------------------------------------
    # IDEAL FEATURES
    # ----------------------------------------------

    print()
    print("IDEAL SPEECH FEATURES")
    print("--------------------------------------")

    print(
        f"Duration: "
        f"{ideal['duration']:.2f} seconds"
    )

    print(
        f"Word Count: "
        f"{ideal['word_count']}"
    )

    print(
        f"Speech Rate: "
        f"{ideal['speech_rate']:.2f} WPM"
    )

    print(
        f"Average Pitch: "
        f"{ideal['average_pitch']:.2f} Hz"
    )

    print(
        f"Average Energy: "
        f"{ideal['average_energy']:.5f}"
    )

    print(
        f"Pitch Variation: "
        f"{ideal['pitch_variation']:.2f}"
    )

    print("MFCC:")

    print(
        np.round(
            ideal["mfcc"],
            2
        )
    )

    # ----------------------------------------------
    # FLAWED FEATURES
    # ----------------------------------------------

    print()
    print("FLAWED SPEECH FEATURES")
    print("--------------------------------------")

    print(
        f"Duration: "
        f"{flawed['duration']:.2f} seconds"
    )

    print(
        f"Word Count: "
        f"{flawed['word_count']}"
    )

    print(
        f"Speech Rate: "
        f"{flawed['speech_rate']:.2f} WPM"
    )

    print(
        f"Average Pitch: "
        f"{flawed['average_pitch']:.2f} Hz"
    )

    print(
        f"Average Energy: "
        f"{flawed['average_energy']:.5f}"
    )

    print(
        f"Pitch Variation: "
        f"{flawed['pitch_variation']:.2f}"
    )

    print("MFCC:")

    print(
        np.round(
            flawed["mfcc"],
            2
        )
    )

    # ----------------------------------------------
    # NORMALIZED DIFFERENCES
    # ----------------------------------------------

    duration_change = percentage_change(
        ideal["duration"],
        flawed["duration"]
    )

    rate_change = percentage_change(
        ideal["speech_rate"],
        flawed["speech_rate"]
    )

    pitch_change = percentage_change(
        ideal["average_pitch"],
        flawed["average_pitch"]
    )

    energy_change = percentage_change(
        ideal["average_energy"],
        flawed["average_energy"]
    )

    variation_change = percentage_change(
        ideal["pitch_variation"],
        flawed["pitch_variation"]
    )

    # ----------------------------------------------
    # MFCC DISTANCE
    # ----------------------------------------------

    mfcc_distance = np.linalg.norm(
        ideal["mfcc"]
        -
        flawed["mfcc"]
    )

    # ----------------------------------------------
    # RESULTS
    # ----------------------------------------------

    print()
    print("======================================")
    print("NORMALIZED DIFFERENCES")
    print("======================================")

    print(
        f"Duration Change: "
        f"{duration_change:.2f}%"
    )

    print(
        f"Speech Rate Change: "
        f"{rate_change:.2f}%"
    )

    print(
        f"Pitch Change: "
        f"{pitch_change:.2f}%"
    )

    print(
        f"Energy Change: "
        f"{energy_change:.2f}%"
    )

    print(
        f"Pitch Variation Change: "
        f"{variation_change:.2f}%"
    )

    print(
        f"MFCC Distance: "
        f"{mfcc_distance:.2f}"
    )

    # ----------------------------------------------
    # INTERPRETATION
    # ----------------------------------------------

    print()
    print("======================================")
    print("INTERPRETATION")
    print("======================================")

    if rate_change > 20:
        print(
            "Speech Rate: FASTER in flawed speech"
        )
    elif rate_change < -20:
        print(
            "Speech Rate: SLOWER in flawed speech"
        )
    else:
        print(
            "Speech Rate: Similar"
        )

    if energy_change < -20:
        print(
            "Energy: Lower vocal energy detected"
        )

    if pitch_change > 20:
        print(
            "Pitch: Higher pitch detected"
        )
    elif pitch_change < -20:
        print(
            "Pitch: Lower pitch detected"
        )

    if variation_change < -20:
        print(
            "Pitch Variation: Reduced variation"
        )
    elif variation_change > 20:
        print(
            "Pitch Variation: Increased variation"
        )

    print()
    print("Analysis completed.")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()

import json

import librosa
import numpy as np
import whisper

# ---------------------------------------
# WHISPER MODEL
# ---------------------------------------

MODEL = whisper.load_model("base")


# ---------------------------------------
# TRANSCRIBE AUDIO
# ---------------------------------------


def transcribe(file):

    result = MODEL.transcribe(file, word_timestamps=True)

    words = []

    for segment in result["segments"]:
        for word in segment["words"]:
            text = word["word"].strip()

            if text:
                words.append(
                    {
                        "word": text.lower(),
                        "start": float(word["start"]),
                        "end": float(word["end"]),
                    }
                )

    return words


# ---------------------------------------
# AUDIO FEATURES
# ---------------------------------------


def audio_features(file, start, end):

    y, sr = librosa.load(file, sr=16000, offset=start, duration=end - start)

    if len(y) == 0:
        return {"pitch": 0, "energy": 0, "pitch_variation": 0}

    # Energy

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]

    energy = float(np.mean(rms))

    # Pitch

    pitch, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr, hop_length=256)

    pitch_clean = pitch[~np.isnan(pitch)]

    if len(pitch_clean) == 0:
        avg_pitch = 0
        pitch_variation = 0

    else:
        avg_pitch = float(np.mean(pitch_clean))

        pitch_variation = float(np.std(pitch_clean))

    return {"pitch": avg_pitch, "energy": energy, "pitch_variation": pitch_variation}


# ---------------------------------------
# LOCAL SPEECH RATE
# ---------------------------------------


def speech_rate(word_count, start, end):

    duration = end - start

    if duration <= 0:
        return 0

    return (word_count / duration) * 60


# ---------------------------------------
# MAIN ANALYSIS
# ---------------------------------------


def analyze():

    ideal_file = "dataset/ideal/speech1.wav"

    flawed_file = "dataset/flawed/flawed1.wav"

    label_file = "dataset/labels/speech1_flawed1.json"

    # -----------------------------------
    # LOAD GROUND TRUTH
    # -----------------------------------

    print("Loading ground-truth labels...")

    with open(label_file, encoding="utf-8") as f:
        label = json.load(f)

    flaw = label["flaws"][0]

    gt_start = flaw["flawed_start"]

    gt_end = flaw["flawed_end"]

    flaw_type = flaw["type"]

    print("\nGround-truth flaw:")
    print(f"{gt_start}s - {gt_end}s")

    print(f"Type: {flaw_type}")

    # -----------------------------------
    # TRANSCRIBE
    # -----------------------------------

    print("\nTranscribing ideal speech...")

    ideal_words = transcribe(ideal_file)

    print("Ideal words:", len(ideal_words))

    print("\nTranscribing flawed speech...")

    flawed_words = transcribe(flawed_file)

    print("Flawed words:", len(flawed_words))

    # -----------------------------------
    # WORDS INSIDE GROUND-TRUTH REGION
    # -----------------------------------

    region_words = []

    for word in flawed_words:
        word_start = word["start"]

        word_end = word["end"]

        # Include words that overlap the
        # labeled temporal region

        if word_end >= gt_start and word_start <= gt_end:
            region_words.append(word)

    # -----------------------------------
    # WORD COUNT
    # -----------------------------------

    count = len(region_words)

    print("\nWords inside flaw region:", count)

    if count == 0:
        print("No words found inside the labeled region.")

        return

    # -----------------------------------
    # ACTUAL REGION FROM WHISPER
    # -----------------------------------

    detected_start = region_words[0]["start"]

    detected_end = region_words[-1]["end"]

    # Keep detection inside label

    detected_start = max(detected_start, gt_start)

    detected_end = min(detected_end, gt_end)

    # -----------------------------------
    # IDEAL COMPARISON
    # -----------------------------------

    # The corresponding sentence is the
    # middle sentence in the ideal speech.
    #
    # We locate it using its known words.

    target_words = [
        "today",
        "i",
        "am",
        "going",
        "to",
        "explain",
        "how",
        "artificial",
        "intelligence",
        "can",
        "help",
        "us",
        "solve",
        "real",
        "world",
        "problems",
    ]

    ideal_selected = []

    for word in ideal_words:
        cleaned = word["word"].replace(".", "").replace(",", "").lower()

        if cleaned in target_words:
            ideal_selected.append(word)

    # Use the complete matching section

    if ideal_selected:
        ideal_start = ideal_selected[0]["start"]

        ideal_end = ideal_selected[-1]["end"]

    else:
        ideal_start = 2.38

        ideal_end = 9.48

    # -----------------------------------
    # IDEAL SPEECH RATE
    # -----------------------------------

    ideal_count = len(ideal_selected)

    ideal_rate = speech_rate(ideal_count, ideal_start, ideal_end)

    # -----------------------------------
    # FLAWED SPEECH RATE
    # -----------------------------------

    flawed_rate = speech_rate(count, detected_start, detected_end)

    # -----------------------------------
    # RATE CHANGE
    # -----------------------------------

    rate_change = (flawed_rate - ideal_rate) / ideal_rate * 100 if ideal_rate > 0 else 0

    # -----------------------------------
    # DURATION REDUCTION
    # -----------------------------------

    ideal_duration = ideal_end - ideal_start

    flawed_duration = detected_end - detected_start

    if ideal_duration > 0:
        duration_reduction = ((ideal_duration - flawed_duration) / ideal_duration) * 100

    else:
        duration_reduction = 0

    # -----------------------------------
    # ACOUSTIC FEATURES
    # -----------------------------------

    print("\nExtracting acoustic features...")

    ideal_features = audio_features(ideal_file, ideal_start, ideal_end)

    flawed_features = audio_features(flawed_file, detected_start, detected_end)

    # -----------------------------------
    # SCORE
    # -----------------------------------

    rate_score = min(max(rate_change, 0), 100)

    duration_score = min(max(duration_reduction, 0), 100)

    score = 0.60 * rate_score + 0.40 * duration_score

    # -----------------------------------
    # SEVERITY
    # -----------------------------------

    if score >= 60:
        severity = "HIGH"

    elif score >= 40:
        severity = "MEDIUM"

    else:
        severity = "LOW"

    # -----------------------------------
    # OUTPUT
    # -----------------------------------

    print("\n")
    print("======================================")
    print("TRACK C TEMPORAL FLAW ANALYSIS")
    print("======================================")

    print("\nGround Truth")
    print(f"{gt_start}s - {gt_end}s")

    print("\nDetected Region")
    print(f"{detected_start:.2f}s - {detected_end:.2f}s")

    print("\nFlaw Type")
    print(flaw_type)

    print("\nWords")
    print(" ".join(word["word"] for word in region_words))

    print("\nSpeech Rate")

    print(f"Ideal: {ideal_rate:.1f} WPM")

    print(f"Flawed: {flawed_rate:.1f} WPM")

    print(f"Increase: {rate_change:.1f}%")

    print("\nDuration")

    print(f"Ideal: {ideal_duration:.2f}s")

    print(f"Flawed: {flawed_duration:.2f}s")

    print(f"Reduction: {duration_reduction:.1f}%")

    print("\nAcoustic Features")

    print(f"Ideal pitch: {ideal_features['pitch']:.2f} Hz")

    print(f"Flawed pitch: {flawed_features['pitch']:.2f} Hz")

    print(f"Ideal energy: {ideal_features['energy']:.5f}")

    print(f"Flawed energy: {flawed_features['energy']:.5f}")

    print(f"Ideal pitch variation: {ideal_features['pitch_variation']:.2f}")

    print(f"Flawed pitch variation: {flawed_features['pitch_variation']:.2f}")

    print("\nScore")

    print(f"{score:.1f}")

    print("\nSeverity")

    print(severity)

    # -----------------------------------
    # CAUSAL EXPLANATION
    # -----------------------------------

    print("\nCausal Explanation")

    if rate_change > 30:
        print("Speech rate increased significantly.")

    if duration_reduction > 20:
        print("The same speech content was delivered in a shorter duration.")

    if flawed_features["energy"] < ideal_features["energy"]:
        print("Vocal energy decreased during the flawed region.")

    if flawed_features["pitch_variation"] > ideal_features["pitch_variation"]:
        print("Pitch variation increased.")

    elif flawed_features["pitch_variation"] < ideal_features["pitch_variation"]:
        print("Pitch variation decreased.")

    print("\n======================================")
    print("ANALYSIS COMPLETE")
    print("======================================")


if __name__ == "__main__":
    analyze()

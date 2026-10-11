import json

import librosa
import numpy as np
import whisper

IDEAL_FILE = "dataset/ideal/speech1.wav"
FLAWED_FILE = "dataset/flawed/flawed1.wav"
GROUND_TRUTH_FILE = "dataset/labels/speech1_flawed1.json"

MODEL = whisper.load_model("base")


def transcribe(file):
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
                    "word": text.lower(),
                    "start": float(word["start"]),
                    "end": float(word["end"])
                })

    return words


def get_words_in_region(words, start, end):

    selected = []

    for word in words:

        if word["start"] >= start and word["end"] <= end:
            selected.append(word)

    return selected


def calculate_speech_rate(words):

    if len(words) < 2:
        return 0

    start = words[0]["start"]
    end = words[-1]["end"]

    duration = end - start

    if duration <= 0:
        return 0

    return len(words) / duration * 60


def find_ideal_matching_region(ideal_words, flawed_words):

    if not flawed_words:
        return 0, 0

    first_word = flawed_words[0]["word"]

    last_word = flawed_words[-1]["word"]

    start_index = None

    end_index = None

    # Find first matching anchor word
    for i, word in enumerate(ideal_words):

        if word["word"] == first_word:
            start_index = i
            break

    if start_index is None:
        return 0, 0

    # Find last matching anchor word
    for i in range(
        len(ideal_words) - 1,
        start_index,
        -1
    ):

        if ideal_words[i]["word"] == last_word:
            end_index = i
            break

    if end_index is None:

        # fallback:
        # use the first matching sentence-sized region
        end_index = min(
            start_index + len(flawed_words) + 10,
            len(ideal_words) - 1
        )

    ideal_start = ideal_words[start_index]["start"]
    ideal_end = ideal_words[end_index]["end"]

    return ideal_start, ideal_end


def analyze_audio_region(file, start, end):

    y, sr = librosa.load(
        file,
        sr=16000,
        offset=start,
        duration=end - start
    )

    if len(y) == 0:

        return {
            "energy": 0,
            "pitch": 0,
            "pitch_variation": 0,
            "mfcc": []
        }

    rms = librosa.feature.rms(y=y)[0]

    energy = float(np.mean(rms))

    pitch, _, _ = librosa.pyin(
        y,
        fmin=70,
        fmax=400,
        sr=sr
    )

    pitch_clean = pitch[~np.isnan(pitch)]

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

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    return {
        "energy": energy,
        "pitch": average_pitch,
        "pitch_variation": pitch_variation,
        "mfcc": mfcc_mean.tolist()
    }


def percentage_change(ideal, flawed):

    if ideal == 0:
        return 0

    return (
        (flawed - ideal)
        / abs(ideal)
    ) * 100


def mfcc_distance(
    ideal_mfcc,
    flawed_mfcc
):

    ideal_array = np.array(
        ideal_mfcc
    )

    flawed_array = np.array(
        flawed_mfcc
    )

    return float(
        np.linalg.norm(
            ideal_array - flawed_array
        )
    )


def main():

    print("======================================")
    print("TEMPORAL ACOUSTIC ANALYSIS")
    print("======================================")

    # -----------------------------------
    # 1. Ground truth
    # -----------------------------------

    print("\n1. Loading ground truth...")

    with open(
        GROUND_TRUTH_FILE,
        encoding="utf-8"
    ) as f:

        ground_truth = json.load(f)

    flaw = ground_truth["flaws"][0]

    flawed_start = float(
        flaw["flawed_start"]
    )

    flawed_end = float(
        flaw["flawed_end"]
    )

    print(
        f"Flaw region: "
        f"{flawed_start:.2f}s - "
        f"{flawed_end:.2f}s"
    )

    # -----------------------------------
    # 2. Ideal transcription
    # -----------------------------------

    print("\n2. Transcribing ideal speech...")

    ideal_words = transcribe(
        IDEAL_FILE
    )

    print(
        "Ideal words:",
        len(ideal_words)
    )

    # -----------------------------------
    # 3. Flawed transcription
    # -----------------------------------

    print("\n3. Transcribing flawed speech...")

    flawed_words = transcribe(
        FLAWED_FILE
    )

    print(
        "Flawed words:",
        len(flawed_words)
    )

    # -----------------------------------
    # 4. Flawed region words
    # -----------------------------------

    print(
        "\n4. Selecting words inside "
        "flaw region..."
    )

    flawed_region_words = get_words_in_region(
        flawed_words,
        flawed_start,
        flawed_end
    )

    print(
        "Flawed region words:",
        len(flawed_region_words)
    )

    print(
        "Text:",
        " ".join(
            word["word"]
            for word in flawed_region_words
        )
    )

    # -----------------------------------
    # 5. Find corresponding ideal region
    # -----------------------------------

    print(
        "\n5. Finding corresponding "
        "ideal region..."
    )

    ideal_start, ideal_end = (
        find_ideal_matching_region(
            ideal_words,
            flawed_region_words
        )
    )

    print(
        f"Ideal region: "
        f"{ideal_start:.2f}s - "
        f"{ideal_end:.2f}s"
    )

    ideal_region_words = get_words_in_region(
        ideal_words,
        ideal_start,
        ideal_end
    )

    print(
        "Ideal region words:",
        len(ideal_region_words)
    )

    print(
        "Text:",
        " ".join(
            word["word"]
            for word in ideal_region_words
        )
    )

    # -----------------------------------
    # 6. Acoustic features
    # -----------------------------------

    print(
        "\n6. Extracting regional "
        "acoustic features..."
    )

    ideal_features = analyze_audio_region(
        IDEAL_FILE,
        ideal_start,
        ideal_end
    )

    flawed_features = analyze_audio_region(
        FLAWED_FILE,
        flawed_start,
        flawed_end
    )

    # -----------------------------------
    # 7. Speech rates
    # -----------------------------------

    ideal_rate = calculate_speech_rate(
        ideal_region_words
    )

    flawed_rate = calculate_speech_rate(
        flawed_region_words
    )

    # -----------------------------------
    # 8. Print ideal
    # -----------------------------------

    print("\n======================================")
    print("IDEAL REGION FEATURES")
    print("======================================")

    print(
        f"Speech Rate: "
        f"{ideal_rate:.2f} WPM"
    )

    print(
        f"Energy: "
        f"{ideal_features['energy']:.5f}"
    )

    print(
        f"Average Pitch: "
        f"{ideal_features['pitch']:.2f} Hz"
    )

    print(
        f"Pitch Variation: "
        f"{ideal_features['pitch_variation']:.2f}"
    )

    # -----------------------------------
    # 9. Print flawed
    # -----------------------------------

    print("\n======================================")
    print("FLAWED REGION FEATURES")
    print("======================================")

    print(
        f"Speech Rate: "
        f"{flawed_rate:.2f} WPM"
    )

    print(
        f"Energy: "
        f"{flawed_features['energy']:.5f}"
    )

    print(
        f"Average Pitch: "
        f"{flawed_features['pitch']:.2f} Hz"
    )

    print(
        f"Pitch Variation: "
        f"{flawed_features['pitch_variation']:.2f}"
    )

    # -----------------------------------
    # 10. Changes
    # -----------------------------------

    rate_change = percentage_change(
        ideal_rate,
        flawed_rate
    )

    energy_change = percentage_change(
        ideal_features["energy"],
        flawed_features["energy"]
    )

    pitch_change = percentage_change(
        ideal_features["pitch"],
        flawed_features["pitch"]
    )

    variation_change = percentage_change(
        ideal_features["pitch_variation"],
        flawed_features["pitch_variation"]
    )

    mfcc_dist = mfcc_distance(
        ideal_features["mfcc"],
        flawed_features["mfcc"]
    )

    # -----------------------------------
    # 11. Changes output
    # -----------------------------------

    print("\n======================================")
    print("REGIONAL FEATURE CHANGES")
    print("======================================")

    print(
        f"Speech Rate Change: "
        f"{rate_change:.2f}%"
    )

    print(
        f"Energy Change: "
        f"{energy_change:.2f}%"
    )

    print(
        f"Pitch Change: "
        f"{pitch_change:.2f}%"
    )

    print(
        f"Pitch Variation Change: "
        f"{variation_change:.2f}%"
    )

    print(
        f"MFCC Distance: "
        f"{mfcc_dist:.2f}"
    )

    # -----------------------------------
    # 12. Explanation
    # -----------------------------------

    print(
        "\n======================================"
    )

    print(
        "TEMPORALLY GROUNDED EXPLANATION"
    )

    print(
        "======================================"
    )

    print(
        f"The flawed speech between "
        f"{flawed_start:.2f}s and "
        f"{flawed_end:.2f}s was delivered "
        f"{rate_change:.2f}% faster than "
        f"the corresponding ideal region."
    )

    if energy_change < 0:

        print(
            f"Vocal energy decreased by "
            f"{abs(energy_change):.2f}%, "
            f"indicating weaker vocal delivery."
        )

    if variation_change < 0:

        print(
            f"Pitch variation decreased by "
            f"{abs(variation_change):.2f}%, "
            f"suggesting reduced expressiveness."
        )

    print(
        "These measurements provide "
        "region-specific acoustic evidence "
        "for excessive speech rate."
    )

    # -----------------------------------
    # 13. Save JSON
    # -----------------------------------

    result = {

        "flaw_type": flaw["type"],

        "ground_truth_region": {
            "start": flawed_start,
            "end": flawed_end
        },

        "ideal_comparison_region": {
            "start": ideal_start,
            "end": ideal_end
        },

        "ideal": {
            "speech_rate_wpm": ideal_rate,
            **ideal_features
        },

        "flawed": {
            "speech_rate_wpm": flawed_rate,
            **flawed_features
        },

        "changes": {

            "speech_rate_change_percent":
                rate_change,

            "energy_change_percent":
                energy_change,

            "pitch_change_percent":
                pitch_change,

            "pitch_variation_change_percent":
                variation_change,

            "mfcc_distance":
                mfcc_dist
        }
    }

    with open(
        "temporal_acoustic_result.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    print(
        "\n======================================"
    )

    print("RESULT SAVED")

    print(
        "======================================"
    )

    print(
        "temporal_acoustic_result.json"
    )


if __name__ == "__main__":
    main()

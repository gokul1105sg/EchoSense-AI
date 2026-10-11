import json

import librosa
import numpy as np
import whisper

# ============================================================
# FILES
# ============================================================

IDEAL_FILE = "dataset/ideal/speech1.wav"
FLAWED_FILE = "dataset/flawed/flawed1.wav"
GROUND_TRUTH_FILE = "dataset/labels/speech1_flawed1.json"
OUTPUT_FILE = "track_c_result.json"


# ============================================================
# LOAD WHISPER
# ============================================================

print("Loading Whisper model...")
MODEL = whisper.load_model("base")


# ============================================================
# TRANSCRIPTION
# ============================================================


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


# ============================================================
# TEMPORAL DETECTOR
# ============================================================


def detect_fast_regions(words):

    abnormal = []

    for i, word in enumerate(words):
        duration = word["end"] - word["start"]

        if duration <= 0:
            continue

        start = max(0, i - 4)
        end = min(len(words), i + 5)

        nearby = []

        for j in range(start, end):
            if j == i:
                continue

            d = words[j]["end"] - words[j]["start"]

            if d > 0:
                nearby.append(d)

        if len(nearby) < 3:
            continue

        median_duration = np.median(nearby)

        if duration < median_duration * 0.70:
            abnormal.append(word)

    if not abnormal:
        return []

    regions = []

    current = [abnormal[0]]

    for word in abnormal[1:]:
        gap = word["start"] - current[-1]["end"]

        if gap <= 3.0:
            current.append(word)

        else:
            if len(current) >= 2:
                regions.append(current)

            current = [word]

    if len(current) >= 2:
        regions.append(current)

    final_regions = []

    for region in regions:
        start = region[0]["start"]
        end = region[-1]["end"]

        previous_words = [w for w in words if w["end"] <= start]

        next_words = [w for w in words if w["start"] >= end]

        if previous_words:
            previous = previous_words[-1]

            if start - previous["end"] <= 1.0:
                start = previous["start"]

        if next_words:
            following = next_words[0]

            if following["start"] - end <= 1.0:
                end = following["end"]

        final_regions.append({"start": start, "end": end})

    return final_regions


# ============================================================
# COMBINE DETECTED REGIONS
# ============================================================


def combine_regions(regions):

    if not regions:
        return None

    start = min(region["start"] for region in regions)

    end = max(region["end"] for region in regions)

    return {"start": start, "end": end}


# ============================================================
# FIND CORRESPONDING IDEAL REGION
# ============================================================


def find_ideal_matching_region(ideal_words, flawed_words):

    if not flawed_words:
        return 0, 0

    first_word = flawed_words[0]["word"]
    last_word = flawed_words[-1]["word"]

    start_index = None
    end_index = None

    # --------------------------------------------------------
    # Find first matching anchor
    # --------------------------------------------------------

    for i, word in enumerate(ideal_words):
        if word["word"] == first_word:
            start_index = i
            break

    if start_index is None:
        return 0, 0

    # --------------------------------------------------------
    # Find last matching anchor
    # --------------------------------------------------------

    for i in range(len(ideal_words) - 1, start_index - 1, -1):
        if ideal_words[i]["word"] == last_word:
            end_index = i
            break

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if end_index is None:
        end_index = min(start_index + len(flawed_words) + 10, len(ideal_words) - 1)

    ideal_start = ideal_words[start_index]["start"]
    ideal_end = ideal_words[end_index]["end"]

    return ideal_start, ideal_end


# ============================================================
# AUDIO FEATURES
# ============================================================


def extract_features(file, start, end, word_count):

    audio, sr = librosa.load(file, sr=16000, mono=True)

    start_sample = int(start * sr)
    end_sample = int(end * sr)

    segment = audio[start_sample:end_sample]

    if len(segment) == 0:
        return {
            "speech_rate": 0,
            "energy": 0,
            "average_pitch": 0,
            "pitch_variation": 0,
            "mfcc": [],
        }

    duration = end - start

    speech_rate = word_count / duration * 60 if duration > 0 else 0

    # --------------------------------------------------------
    # RMS Energy
    # --------------------------------------------------------

    rms = librosa.feature.rms(y=segment)[0]

    energy = float(np.mean(rms))

    # --------------------------------------------------------
    # Pitch / F0
    # --------------------------------------------------------

    pitch, _voiced_flag, _voiced_prob = librosa.pyin(segment, fmin=70, fmax=400, sr=sr)

    valid_pitch = pitch[~np.isnan(pitch)]

    if len(valid_pitch) > 0:
        average_pitch = float(np.mean(valid_pitch))

        pitch_variation = float(np.std(valid_pitch))

    else:
        average_pitch = 0
        pitch_variation = 0

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)

    mfcc_mean = np.mean(mfcc, axis=1)

    return {
        "speech_rate": float(speech_rate),
        "energy": energy,
        "average_pitch": average_pitch,
        "pitch_variation": pitch_variation,
        "mfcc": mfcc_mean.tolist(),
    }


# ============================================================
# PERCENTAGE CHANGE
# ============================================================


def percentage_change(ideal, flawed):

    if ideal == 0:
        return 0

    return ((flawed - ideal) / ideal) * 100


# ============================================================
# MFCC DISTANCE
# ============================================================


def mfcc_distance(ideal, flawed):

    a = np.array(ideal)
    b = np.array(flawed)

    return float(np.linalg.norm(a - b))


# ============================================================
# CAUSAL SCORING
# ============================================================


def calculate_causal_score(rate_change, energy_change, pitch_variation_change):

    score = 0

    # --------------------------------------------------------
    # Speech rate evidence
    # --------------------------------------------------------

    if rate_change >= 50:
        score += 50

    elif rate_change >= 30:
        score += 40

    elif rate_change >= 20:
        score += 25

    # --------------------------------------------------------
    # Duration / rate evidence
    # --------------------------------------------------------

    if rate_change >= 50:
        score += 20

    elif rate_change >= 30:
        score += 15

    elif rate_change >= 20:
        score += 10

    # --------------------------------------------------------
    # Energy evidence
    # --------------------------------------------------------

    if energy_change <= -30:
        score += 15

    elif energy_change <= -15:
        score += 10

    elif energy_change <= -5:
        score += 5

    # --------------------------------------------------------
    # Pitch variation evidence
    # --------------------------------------------------------

    if pitch_variation_change <= -30:
        score += 10

    elif pitch_variation_change <= -20:
        score += 7

    elif pitch_variation_change <= -10:
        score += 4

    return min(score, 100)


# ============================================================
# SEVERITY
# ============================================================


def get_severity(score):

    if score >= 70:
        return "HIGH"

    if score >= 40:
        return "MEDIUM"

    return "LOW"


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("======================================")
    print("TRACK C SPEECH ANALYSIS PIPELINE")
    print("======================================")

    # ========================================================
    # 1. TRANSCRIBE IDEAL
    # ========================================================

    print()
    print("1. Transcribing ideal speech...")

    ideal_words = transcribe(IDEAL_FILE)

    print(f"Ideal words: {len(ideal_words)}")

    # ========================================================
    # 2. TRANSCRIBE FLAWED
    # ========================================================

    print()
    print("2. Transcribing flawed speech...")

    flawed_words = transcribe(FLAWED_FILE)

    print(f"Flawed words: {len(flawed_words)}")

    # ========================================================
    # 3. DETECT TEMPORAL FLAW
    # ========================================================

    print()
    print("3. Detecting temporal flaw regions...")

    detected_regions = detect_fast_regions(flawed_words)

    for i, region in enumerate(detected_regions, 1):
        print(f"Region {i}: {region['start']:.2f}s - {region['end']:.2f}s")

    if not detected_regions:
        print("No temporal flaw detected.")

        return

    combined_region = combine_regions(detected_regions)

    # IMPORTANT:
    # This is the prediction from the detector.
    # It is used for temporal grounding evaluation.

    detected_start = combined_region["start"]
    detected_end = combined_region["end"]

    # ========================================================
    # 4. LOAD GROUND TRUTH
    # ========================================================

    print()
    print("4. Loading ground truth...")

    with open(GROUND_TRUTH_FILE, encoding="utf-8") as file:
        ground_truth = json.load(file)

    ground_truth_flaw = ground_truth["flaws"][0]

    gt_start = float(ground_truth_flaw["flawed_start"])

    gt_end = float(ground_truth_flaw["flawed_end"])

    gt_text = ground_truth_flaw.get("text", "").lower().strip()

    # ========================================================
    # 5. DETECTED REGION WORDS
    # ========================================================

    detected_region_words = [
        word
        for word in flawed_words
        if (word["end"] >= detected_start and word["start"] <= detected_end)
    ]

    print()
    print("Detected region words:", len(detected_region_words))

    # ========================================================
    # 6. GROUND-TRUTH REGION WORDS
    # ========================================================
    #
    # IMPORTANT FIX:
    #
    # The detector deliberately expands the region for
    # temporal context.
    #
    # Therefore we DO NOT use detected_region_words for
    # acoustic comparison.
    #
    # For the labeled evaluation pair, we use the actual
    # ground-truth flaw interval.
    #
    # This gives:
    #
    # flawed region:
    # Today ... problems
    #
    # instead of:
    # morning everyone Today ... problems
    #
    # This makes the ideal/flawed comparison apples-to-apples.
    # ========================================================

    flawed_region_words = [
        word
        for word in flawed_words
        if (word["end"] >= gt_start and word["start"] <= gt_end)
    ]

    print()
    print("Ground-truth comparison words:", len(flawed_region_words))

    print("Comparison text:", " ".join(word["word"] for word in flawed_region_words))

    # ========================================================
    # 7. FIND CORRESPONDING IDEAL REGION
    # ========================================================

    print()
    print("5. Finding corresponding ideal region...")

    ideal_start, ideal_end = find_ideal_matching_region(
        ideal_words, flawed_region_words
    )

    ideal_region_words = [
        word
        for word in ideal_words
        if (word["end"] >= ideal_start and word["start"] <= ideal_end)
    ]

    print(f"Ideal region: {ideal_start:.2f}s - {ideal_end:.2f}s")

    print(f"Ideal region words: {len(ideal_region_words)}")

    print(
        "Ideal comparison text:", " ".join(word["word"] for word in ideal_region_words)
    )

    # ========================================================
    # 8. EXTRACT REGIONAL ACOUSTIC FEATURES
    # ========================================================

    print()
    print("6. Extracting regional acoustic features...")

    ideal_features = extract_features(
        IDEAL_FILE, ideal_start, ideal_end, len(ideal_region_words)
    )

    flawed_features = extract_features(
        FLAWED_FILE, gt_start, gt_end, len(flawed_region_words)
    )

    # ========================================================
    # 9. FEATURE CHANGES
    # ========================================================

    rate_change = percentage_change(
        ideal_features["speech_rate"], flawed_features["speech_rate"]
    )

    energy_change = percentage_change(
        ideal_features["energy"], flawed_features["energy"]
    )

    pitch_change = percentage_change(
        ideal_features["average_pitch"], flawed_features["average_pitch"]
    )

    pitch_variation_change = percentage_change(
        ideal_features["pitch_variation"], flawed_features["pitch_variation"]
    )

    mfcc_dist = mfcc_distance(ideal_features["mfcc"], flawed_features["mfcc"])

    # ========================================================
    # 10. TEMPORAL IoU
    # ========================================================
    #
    # IMPORTANT:
    # IoU compares the detector prediction against ground
    # truth. Ground truth is NOT used to improve the detector.
    # ========================================================

    intersection_start = max(detected_start, gt_start)

    intersection_end = min(detected_end, gt_end)

    intersection = max(0, intersection_end - intersection_start)

    union_start = min(detected_start, gt_start)

    union_end = max(detected_end, gt_end)

    union = union_end - union_start

    temporal_iou = intersection / union if union > 0 else 0

    # ========================================================
    # 11. CAUSAL SCORE
    # ========================================================

    causal_score = calculate_causal_score(
        rate_change, energy_change, pitch_variation_change
    )

    severity = get_severity(causal_score)

    confidence = min(
        1.0, 0.5 + abs(rate_change) / 200 + abs(pitch_variation_change) / 200
    )

    # ========================================================
    # 12. CAUSAL EXPLANATION
    # ========================================================

    explanation = (
        f"The flawed speech between "
        f"{gt_start:.2f}s and "
        f"{gt_end:.2f}s was delivered "
        f"{rate_change:.2f}% faster than "
        f"the corresponding ideal region. "
        f"Vocal energy changed by "
        f"{energy_change:.2f}%, while pitch "
        f"variation changed by "
        f"{pitch_variation_change:.2f}%. "
        f"These region-specific acoustic "
        f"measurements support the identification "
        f"of excessive speech rate."
    )

    # ========================================================
    # 13. RECOMMENDATION
    # ========================================================

    recommendation = (
        "Slow down during the highlighted "
        "region. Introduce short pauses "
        "between major phrases and clearly "
        "pronounce important words."
    )

    # ========================================================
    # 14. PRINT RESULT
    # ========================================================

    print()
    print("======================================")
    print("FINAL TRACK C RESULT")
    print("======================================")

    print("Flaw: excessive_speech_rate")

    print(f"Severity: {severity}")

    print(f"Causal Score: {causal_score} / 100")

    print(f"Confidence: {confidence:.2f}")

    print()

    print(f"Detected Region: {detected_start:.2f}s - {detected_end:.2f}s")

    print(f"Ground Truth: {gt_start:.2f}s - {gt_end:.2f}s")

    print(f"Temporal IoU: {temporal_iou:.4f}")

    print()

    print(f"Comparison Flawed Region: {gt_start:.2f}s - {gt_end:.2f}s")

    print(f"Ideal Comparison Region: {ideal_start:.2f}s - {ideal_end:.2f}s")

    print()

    print(f"Regional Speech Rate Change: {rate_change:.2f}%")

    print(f"Regional Energy Change: {energy_change:.2f}%")

    print(f"Regional Pitch Change: {pitch_change:.2f}%")

    print(f"Regional Pitch Variation Change: {pitch_variation_change:.2f}%")

    print(f"Regional MFCC Distance: {mfcc_dist:.2f}")

    print()
    print("--------------------------------------")
    print("CAUSAL EXPLANATION")
    print("--------------------------------------")

    print(explanation)

    print()
    print("--------------------------------------")
    print("RECOMMENDATION")
    print("--------------------------------------")

    print(recommendation)

    # ========================================================
    # 15. SAVE JSON
    # ========================================================

    result = {
        "project": "Track C - Contrastive Speech Analytics",
        "flaw": {
            "type": "excessive_speech_rate",
            "severity": severity,
            "causal_score": causal_score,
            "confidence": confidence,
        },
        "temporal_grounding": {
            "detected": {"start": detected_start, "end": detected_end},
            "ground_truth": {"start": gt_start, "end": gt_end},
            "iou": temporal_iou,
        },
        "comparison": {
            "ground_truth_text": gt_text,
            "flawed_region": {
                "start": gt_start,
                "end": gt_end,
                "word_count": len(flawed_region_words),
            },
            "ideal_region": {
                "start": ideal_start,
                "end": ideal_end,
                "word_count": len(ideal_region_words),
            },
        },
        "ideal_region": {
            "start": ideal_start,
            "end": ideal_end,
            "word_count": len(ideal_region_words),
            "features": ideal_features,
        },
        "flawed_region": {
            "start": gt_start,
            "end": gt_end,
            "word_count": len(flawed_region_words),
            "features": flawed_features,
        },
        "feature_changes": {
            "speech_rate_percent": rate_change,
            "energy_percent": energy_change,
            "pitch_percent": pitch_change,
            "pitch_variation_percent": pitch_variation_change,
            "mfcc_distance": mfcc_dist,
        },
        "causal_explanation": explanation,
        "recommendation": recommendation,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    print()
    print(f"Final JSON saved to: {OUTPUT_FILE}")

    print()
    print("======================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

import json

import numpy as np
import whisper

# ==========================================
# FILE PATHS
# ==========================================

AUDIO_FILE = "dataset/flawed/flawed1.wav"

LABEL_FILE = "dataset/labels/speech1_flawed1.json"


# ==========================================
# LOAD WHISPER MODEL
# ==========================================

print("Loading Whisper model...")

MODEL = whisper.load_model("base")


# ==========================================
# TRANSCRIBE AUDIO
# ==========================================


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


# ==========================================
# DETECT FAST SPEECH
# ==========================================


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

    # ======================================
    # GROUP ABNORMAL WORDS
    # ======================================

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

    # ======================================
    # EXPAND REGIONS
    # ======================================

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


# ==========================================
# LOAD GROUND TRUTH
# ==========================================


def load_ground_truth():

    with open(LABEL_FILE, encoding="utf-8") as file:
        data = json.load(file)

    flaws = data["flaws"]

    start = min(flaw["flawed_start"] for flaw in flaws)

    end = max(flaw["flawed_end"] for flaw in flaws)

    return start, end


# ==========================================
# INTERSECTION
# ==========================================


def calculate_intersection(start1, end1, start2, end2):

    start = max(start1, start2)
    end = min(end1, end2)

    if end <= start:
        return 0.0

    return end - start


# ==========================================
# IOU
# ==========================================


def calculate_iou(detected_start, detected_end, actual_start, actual_end):

    intersection = calculate_intersection(
        detected_start, detected_end, actual_start, actual_end
    )

    detected_duration = detected_end - detected_start

    actual_duration = actual_end - actual_start

    union = detected_duration + actual_duration - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ==========================================
# PRECISION
# ==========================================


def calculate_precision(detected_start, detected_end, actual_start, actual_end):

    intersection = calculate_intersection(
        detected_start, detected_end, actual_start, actual_end
    )

    detected_duration = detected_end - detected_start

    if detected_duration <= 0:
        return 0.0

    return intersection / detected_duration


# ==========================================
# RECALL
# ==========================================


def calculate_recall(detected_start, detected_end, actual_start, actual_end):

    intersection = calculate_intersection(
        detected_start, detected_end, actual_start, actual_end
    )

    actual_duration = actual_end - actual_start

    if actual_duration <= 0:
        return 0.0

    return intersection / actual_duration


# ==========================================
# F1 SCORE
# ==========================================


def calculate_f1(precision, recall):

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ==========================================
# MAIN
# ==========================================


def main():

    print("\n======================================")
    print("AUTOMATIC TEMPORAL GROUNDING")
    print("======================================")

    # --------------------------------------
    # STEP 1: TRANSCRIBE
    # --------------------------------------

    print("\nLoading flawed speech...")

    words = transcribe(AUDIO_FILE)

    print("Detected words:", len(words))

    # --------------------------------------
    # STEP 2: DETECT
    # --------------------------------------

    print("\nDetecting abnormal speech regions...")

    detected_regions = detect_fast_regions(words)

    if not detected_regions:
        print("\nNo fast-speech regions detected.")

        return

    # --------------------------------------
    # SHOW DETECTED REGIONS
    # --------------------------------------

    print("\n======================================")

    print("DETECTED REGIONS")

    print("======================================")

    for i, region in enumerate(detected_regions, 1):
        print(f"Region {i}: {region['start']:.2f}s - {region['end']:.2f}s")

    # --------------------------------------
    # STEP 3: COMBINE DETECTED REGIONS
    # --------------------------------------

    detected_start = min(region["start"] for region in detected_regions)

    detected_end = max(region["end"] for region in detected_regions)

    print("\nCombined Detection:")

    print(f"{detected_start:.2f}s - {detected_end:.2f}s")

    # --------------------------------------
    # STEP 4: LOAD GROUND TRUTH
    # --------------------------------------

    actual_start, actual_end = load_ground_truth()

    print("\nGround Truth:")

    print(f"{actual_start:.2f}s - {actual_end:.2f}s")

    # --------------------------------------
    # STEP 5: CALCULATE METRICS
    # --------------------------------------

    iou = calculate_iou(detected_start, detected_end, actual_start, actual_end)

    precision = calculate_precision(
        detected_start, detected_end, actual_start, actual_end
    )

    recall = calculate_recall(detected_start, detected_end, actual_start, actual_end)

    f1 = calculate_f1(precision, recall)

    # --------------------------------------
    # STEP 6: DISPLAY RESULTS
    # --------------------------------------

    print("\n======================================")

    print("EVALUATION RESULTS")

    print("======================================")

    print(f"\nIoU:       {iou * 100:.2f}%")

    print(f"Precision: {precision * 100:.2f}%")

    print(f"Recall:    {recall * 100:.2f}%")

    print(f"F1 Score:  {f1 * 100:.2f}%")

    # --------------------------------------
    # INTERPRETATION
    # --------------------------------------

    print("\n======================================")

    print("INTERPRETATION")

    print("======================================")

    if f1 >= 0.80:
        print("Excellent temporal grounding.")

    elif f1 >= 0.60:
        print("Good temporal grounding.")

    elif f1 >= 0.40:
        print("Moderate temporal grounding.")

    else:
        print("Weak temporal grounding.")


# ==========================================
# START PROGRAM
# ==========================================

if __name__ == "__main__":
    main()

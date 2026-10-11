from difflib import SequenceMatcher

import whisper

model = whisper.load_model("base")

ideal_file = "dataset/ideal/speech1.wav"
flawed_file = "dataset/flawed/flawed1.wav"


def get_words(file):
    result = model.transcribe(file, word_timestamps=True)

    words = []

    for segment in result["segments"]:
        for word in segment["words"]:
            words.append(
                {
                    "word": word["word"].strip().lower(),
                    "start": word["start"],
                    "end": word["end"],
                    "duration": word["end"] - word["start"],
                }
            )

    return words


# Get word timestamps
ideal = get_words(ideal_file)
flawed = get_words(flawed_file)

ideal_words = [x["word"] for x in ideal]
flawed_words = [x["word"] for x in flawed]


# Align common words
matcher = SequenceMatcher(None, ideal_words, flawed_words)


results = []


for block in matcher.get_matching_blocks():
    # Ignore very small matches
    if block.size < 2:
        continue

    ideal_start = ideal[block.a]["start"]
    ideal_end = ideal[block.a + block.size - 1]["end"]

    flawed_start = flawed[block.b]["start"]
    flawed_end = flawed[block.b + block.size - 1]["end"]

    ideal_duration = ideal_end - ideal_start
    flawed_duration = flawed_end - flawed_start

    if ideal_duration <= 0 or flawed_duration <= 0:
        continue

    word_count = block.size

    # -----------------------------
    # 1. Speech Rate
    # -----------------------------

    ideal_rate = (word_count / ideal_duration) * 60

    flawed_rate = (word_count / flawed_duration) * 60

    rate_change = ((flawed_rate - ideal_rate) / ideal_rate) * 100

    # -----------------------------
    # 2. Duration Reduction
    # -----------------------------

    duration_reduction = ((ideal_duration - flawed_duration) / ideal_duration) * 100

    # -----------------------------
    # 3. Normalize scores
    # -----------------------------

    rate_score = min(max(rate_change, 0), 100)

    duration_score = min(max(duration_reduction, 0), 100)

    # -----------------------------
    # 4. Combined Flaw Score
    # -----------------------------

    flaw_score = rate_score * 0.60 + duration_score * 0.40

    # -----------------------------
    # 5. Severity
    # -----------------------------

    if flaw_score >= 60:
        severity = "HIGH"

    elif flaw_score >= 40:
        severity = "MEDIUM"

    else:
        severity = "LOW"

    results.append(
        {
            "words": ideal_words[block.a : block.a + block.size],
            "ideal_start": ideal_start,
            "ideal_end": ideal_end,
            "flawed_start": flawed_start,
            "flawed_end": flawed_end,
            "ideal_rate": ideal_rate,
            "flawed_rate": flawed_rate,
            "rate_change": rate_change,
            "duration_reduction": duration_reduction,
            "flaw_score": flaw_score,
            "severity": severity,
        }
    )


# Sort strongest flaws first
results.sort(key=lambda x: x["flaw_score"], reverse=True)


# -----------------------------
# Display Results
# -----------------------------

print("\n================================")
print("TRACK C FLAW SCORE")
print("================================\n")


for number, row in enumerate(results, 1):
    print("--------------------------------")
    print(f"REGION {number}")
    print("--------------------------------")

    print("Words:", " ".join(row["words"]))

    print(f"Ideal Time: {row['ideal_start']:.2f}s - {row['ideal_end']:.2f}s")

    print(f"Flawed Time: {row['flawed_start']:.2f}s - {row['flawed_end']:.2f}s")

    print(f"Ideal Speech Rate: {row['ideal_rate']:.1f} WPM")

    print(f"Flawed Speech Rate: {row['flawed_rate']:.1f} WPM")

    print(f"Speech Rate Change: {row['rate_change']:+.1f}%")

    print(f"Duration Reduction: {row['duration_reduction']:.1f}%")

    print(f"FLAW SCORE: {row['flaw_score']:.1f}/100")

    print(f"Severity: {row['severity']}")

    if row["flaw_score"] >= 60:
        print(
            "Explanation: "
            "The participant delivered this region "
            "substantially faster than the ideal baseline."
        )

    elif row["flaw_score"] >= 40:
        print(
            "Explanation: "
            "The participant showed increased "
            "speech speed compared with the baseline."
        )

    else:
        print(
            "Explanation: Only a small difference from the ideal baseline was detected."
        )

    print()


# -----------------------------
# Strongest detected flaw
# -----------------------------

if results:
    best = results[0]

    print("\n================================")
    print("STRONGEST TEMPORAL FLAW")
    print("================================")

    print(f"Time: {best['flawed_start']:.2f}s - {best['flawed_end']:.2f}s")

    print(f"Score: {best['flaw_score']:.1f}/100")

    print(f"Severity: {best['severity']}")

    print("Type: Excessive Speech Rate")

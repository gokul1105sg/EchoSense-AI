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
            words.append({
                "word": word["word"].strip().lower(),
                "start": word["start"],
                "end": word["end"],
                "duration": word["end"] - word["start"]
            })

    return words


ideal = get_words(ideal_file)
flawed = get_words(flawed_file)

ideal_words = [x["word"] for x in ideal]
flawed_words = [x["word"] for x in flawed]

matcher = SequenceMatcher(None, ideal_words, flawed_words)

fast_words = []

for block in matcher.get_matching_blocks():

    if block.size == 0:
        continue

    for i in range(block.size):

        iw = ideal[block.a + i]
        fw = flawed[block.b + i]

        if iw["duration"] == 0:
            continue

        reduction = (
            (iw["duration"] - fw["duration"])
            / iw["duration"]
        ) * 100

        if reduction >= 40:
            fast_words.append({
                "word": iw["word"],
                "start": fw["start"],
                "end": fw["end"],
                "reduction": reduction
            })


print("\n==============================")
print("TEMPORAL FLAW REGIONS")
print("==============================\n")


regions = []

if fast_words:

    start = fast_words[0]["start"]
    end = fast_words[0]["end"]
    words = [fast_words[0]]

    for item in fast_words[1:]:

        gap = item["start"] - end

        # Words within 1.5 seconds belong
        # to the same possible flaw region.
        if gap <= 1.5:

            end = item["end"]
            words.append(item)

        else:

            regions.append({
                "start": start,
                "end": end,
                "words": words
            })

            start = item["start"]
            end = item["end"]
            words = [item]

    regions.append({
        "start": start,
        "end": end,
        "words": words
    })


for region in regions:

    reductions = [
        word["reduction"]
        for word in region["words"]
    ]

    average = sum(reductions) / len(reductions)

    if average >= 60:
        severity = "HIGH"
    elif average >= 40:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    print("--------------------------------")
    print("FLAW DETECTED")
    print("--------------------------------")

    print(
        f"Start: {region['start']:.2f} s"
    )

    print(
        f"End:   {region['end']:.2f} s"
    )

    print(
        f"Duration: "
        f"{region['end'] - region['start']:.2f} s"
    )

    print("Type: Excessive Speech Rate")

    print(f"Severity: {severity}")

    print(
        f"Average duration reduction: "
        f"{average:.1f}%"
    )

    print(
        "Words:",
        ", ".join(
            word["word"]
            for word in region["words"]
        )
    )

    print()

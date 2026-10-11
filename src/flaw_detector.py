import whisper
from difflib import SequenceMatcher

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

results = []

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

        results.append({
            "word": iw["word"],
            "ideal_start": iw["start"],
            "ideal_end": iw["end"],
            "flawed_start": fw["start"],
            "flawed_end": fw["end"],
            "ideal_duration": iw["duration"],
            "flawed_duration": fw["duration"],
            "reduction": reduction
        })


print("\n==============================")
print("TEMPORAL FLAW ANALYSIS")
print("==============================\n")

for row in results:

    if row["reduction"] >= 40:

        print(
            f"⚠ {row['word']:15} | "
            f"Ideal: {row['ideal_duration']:.2f}s | "
            f"Flawed: {row['flawed_duration']:.2f}s | "
            f"Faster: {row['reduction']:.1f}% | "
            f"Time: {row['flawed_start']:.2f}-"
            f"{row['flawed_end']:.2f}s"
        )
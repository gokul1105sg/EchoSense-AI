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


ideal = get_words(ideal_file)
flawed = get_words(flawed_file)

ideal_words = [x["word"] for x in ideal]
flawed_words = [x["word"] for x in flawed]

matcher = SequenceMatcher(None, ideal_words, flawed_words)

print("\n==============================")
print("SPEECH ALIGNMENT")
print("==============================\n")

matches = matcher.get_matching_blocks()

for block in matches:
    if block.size == 0:
        continue

    print(f"\nMatching region: {block.a} - {block.a + block.size}")

    for i in range(block.size):
        iw = ideal[block.a + i]
        fw = flawed[block.b + i]

        print(
            f"{iw['word']:15} | "
            f"Ideal: {iw['start']:.2f}-{iw['end']:.2f}s "
            f"({iw['duration']:.2f}s) | "
            f"Flawed: {fw['start']:.2f}-{fw['end']:.2f}s "
            f"({fw['duration']:.2f}s)"
        )

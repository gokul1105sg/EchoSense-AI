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
                }
            )

    return words


ideal = get_words(ideal_file)
flawed = get_words(flawed_file)

ideal_words = [x["word"] for x in ideal]
flawed_words = [x["word"] for x in flawed]

matcher = SequenceMatcher(None, ideal_words, flawed_words)

print("\n==============================")
print("SPEECH RATE ANALYSIS")
print("==============================\n")


for block in matcher.get_matching_blocks():
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

    ideal_rate = (word_count / ideal_duration) * 60

    flawed_rate = (word_count / flawed_duration) * 60

    rate_change = ((flawed_rate - ideal_rate) / ideal_rate) * 100

    print("--------------------------------")
    print("MATCHED REGION")
    print("--------------------------------")

    print(f"Words: {word_count}")

    print(f"Words: {' '.join(ideal_words[block.a : block.a + block.size])}")

    print(f"Ideal time: {ideal_start:.2f}s - {ideal_end:.2f}s")

    print(f"Flawed time: {flawed_start:.2f}s - {flawed_end:.2f}s")

    print(f"Ideal speech rate: {ideal_rate:.1f} words/min")

    print(f"Flawed speech rate: {flawed_rate:.1f} words/min")

    print(f"Speech-rate change: {rate_change:+.1f}%")

    if rate_change >= 30:
        print("⚠ Possible excessive speech rate")

    print()

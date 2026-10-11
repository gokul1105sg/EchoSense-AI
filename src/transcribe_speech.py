import whisper

model = whisper.load_model("base")


files = {"ideal": "dataset/ideal/speech1.wav", "flawed": "dataset/flawed/flawed1.wav"}


for name, file in files.items():
    print("\n==============================")
    print(name.upper())
    print("==============================")

    result = model.transcribe(file, word_timestamps=True)

    print("\nFull text:")
    print(result["text"])

    print("\nWord timestamps:")

    for segment in result["segments"]:
        for word in segment["words"]:
            print(f"{word['start']:.2f}s - {word['end']:.2f}s : {word['word'].strip()}")

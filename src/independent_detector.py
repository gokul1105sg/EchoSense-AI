import numpy as np
import whisper

# ==========================================
# LOAD WHISPER MODEL
# ==========================================

MODEL = whisper.load_model("base")


# ==========================================
# TRANSCRIBE AUDIO WITH WORD TIMESTAMPS
# ==========================================

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


# ==========================================
# DETECT ABNORMALLY FAST SPEECH
# ==========================================

def detect_fast_regions(words):

    abnormal = []

    # Check every word
    for i, word in enumerate(words):

        duration = word["end"] - word["start"]

        if duration <= 0:
            continue

        # Look at nearby words
        start = max(0, i - 4)
        end = min(len(words), i + 5)

        nearby = []

        for j in range(start, end):

            if j == i:
                continue

            d = words[j]["end"] - words[j]["start"]

            if d > 0:
                nearby.append(d)

        # Need enough nearby words
        if len(nearby) < 3:
            continue

        # Calculate median duration
        median_duration = np.median(nearby)

        # If the word is much shorter than nearby words,
        # mark it as abnormal
        if duration < median_duration * 0.70:

            abnormal.append(word)

    # No abnormal words
    if not abnormal:
        return []

    # ==========================================
    # GROUP ABNORMAL WORDS INTO REGIONS
    # ==========================================

    regions = []

    current = [abnormal[0]]

    for word in abnormal[1:]:

        gap = word["start"] - current[-1]["end"]

        # If abnormal words are close together,
        # consider them part of the same region
        if gap <= 3.0:

            current.append(word)

        else:

            if len(current) >= 2:
                regions.append(current)

            current = [word]

    # Add final group
    if len(current) >= 2:
        regions.append(current)

    # ==========================================
    # EXPAND REGIONS
    # ==========================================

    final_regions = []

    for region in regions:

        start = region[0]["start"]
        end = region[-1]["end"]

        # Find word before the region
        previous_words = [
            w for w in words
            if w["end"] <= start
        ]

        # Find word after the region
        next_words = [
            w for w in words
            if w["start"] >= end
        ]

        # Include previous word if it is close
        if previous_words:

            previous = previous_words[-1]

            if start - previous["end"] <= 1.0:

                start = previous["start"]

        # Include next word if it is close
        if next_words:

            following = next_words[0]

            if following["start"] - end <= 1.0:

                end = following["end"]

        final_regions.append({
            "start": start,
            "end": end
        })

    return final_regions


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    # Flawed speech audio
    file = "dataset/flawed/flawed1.wav"

    print("Loading flawed speech...")

    # Transcribe speech
    words = transcribe(file)

    print("Detected words:", len(words))

    print("\nDetecting abnormal speech regions...")

    # Detect fast speech regions
    regions = detect_fast_regions(words)

    # ==========================================
    # IF NO REGION IS FOUND
    # ==========================================

    if not regions:

        print("\nNo fast-speech regions detected.")

        return

    # ==========================================
    # DISPLAY DETECTED REGIONS
    # ==========================================

    print("\n======================================")
    print("INDEPENDENT TEMPORAL DETECTOR")
    print("======================================")

    for i, region in enumerate(regions, 1):

        start = region["start"]
        end = region["end"]

        duration = end - start

        # Get words inside the region
        region_words = [
            word
            for word in words
            if word["end"] >= start
            and word["start"] <= end
        ]

        # Create text
        text = " ".join(
            word["word"]
            for word in region_words
        )

        # Calculate local speech rate
        rate = len(region_words) / duration * 60 if duration > 0 else 0

        print(f"\nRegion {i}")
        print(f"Time: {start:.2f}s - {end:.2f}s")
        print(f"Words: {text}")
        print(f"Local rate: {rate:.1f} WPM")


# ==========================================
# START PROGRAM
# ==========================================

if __name__ == "__main__":
    main()

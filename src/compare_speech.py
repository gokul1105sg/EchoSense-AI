from temporal_analysis import analyze_temporal

ideal_file = "dataset/ideal/speech1.wav"
flawed_file = "dataset/flawed/flawed1.wav"


ideal = analyze_temporal(ideal_file)
flawed = analyze_temporal(flawed_file)


length = min(len(ideal), len(flawed))

differences = []

for i in range(length):

    energy_diff = abs(
        ideal[i]["energy"] - flawed[i]["energy"]
    )

    pitch_diff = abs(
        ideal[i]["pitch"] - flawed[i]["pitch"]
    )

    score = energy_diff + (pitch_diff / 100)

    differences.append({
        "time": flawed[i]["time"],
        "energy_difference": energy_diff,
        "pitch_difference": pitch_diff,
        "score": score
    })


differences.sort(
    key=lambda x: x["score"],
    reverse=True
)


print("\nTOP TEMPORAL DIFFERENCES")
print("========================\n")

for row in differences[:20]:

    print(
        f"Time: {row['time']} sec | "
        f"Energy Difference: {row['energy_difference']:.5f} | "
        f"Pitch Difference: {row['pitch_difference']:.2f} Hz | "
        f"Score: {row['score']:.4f}"
    )

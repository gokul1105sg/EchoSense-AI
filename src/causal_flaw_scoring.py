import json


# --------------------------------------------------
# FEATURE CHANGE VALUES
# --------------------------------------------------

FEATURES = {
    "duration_change": -40.64,
    "speech_rate_change": 74.66,
    "pitch_change": 5.04,
    "energy_change": -35.86,
    "pitch_variation_change": -23.29,
    "mfcc_distance": 125.06
}


# --------------------------------------------------
# CALCULATE CAUSAL SCORE
# --------------------------------------------------

def calculate_score(features):

    score = 0

    # Speech rate is the strongest indicator
    if features["speech_rate_change"] >= 50:
        score += 50

    elif features["speech_rate_change"] >= 30:
        score += 40

    elif features["speech_rate_change"] >= 20:
        score += 30

    # Duration reduction
    if features["duration_change"] <= -30:
        score += 20

    elif features["duration_change"] <= -20:
        score += 15

    elif features["duration_change"] <= -10:
        score += 10

    # Energy reduction
    if features["energy_change"] <= -30:
        score += 15

    elif features["energy_change"] <= -20:
        score += 10

    # Pitch variation reduction
    if features["pitch_variation_change"] <= -20:
        score += 10

    elif features["pitch_variation_change"] <= -10:
        score += 5

    # MFCC acoustic difference
    if features["mfcc_distance"] >= 100:
        score += 5

    return min(score, 100)


# --------------------------------------------------
# DETERMINE SEVERITY
# --------------------------------------------------

def get_severity(score):

    if score >= 70:
        return "HIGH"

    elif score >= 40:
        return "MEDIUM"

    else:
        return "LOW"


# --------------------------------------------------
# DETERMINE CONFIDENCE
# --------------------------------------------------

def get_confidence(features):

    confidence = 0

    # Strong speech-rate evidence
    if features["speech_rate_change"] >= 50:
        confidence += 0.40

    elif features["speech_rate_change"] >= 30:
        confidence += 0.30

    # Strong duration evidence
    if features["duration_change"] <= -30:
        confidence += 0.25

    elif features["duration_change"] <= -20:
        confidence += 0.20

    # Energy evidence
    if features["energy_change"] <= -30:
        confidence += 0.15

    elif features["energy_change"] <= -20:
        confidence += 0.10

    # Pitch variation evidence
    if features["pitch_variation_change"] <= -20:
        confidence += 0.10

    elif features["pitch_variation_change"] <= -10:
        confidence += 0.05

    # MFCC evidence
    if features["mfcc_distance"] >= 100:
        confidence += 0.10

    return min(confidence, 1.0)


# --------------------------------------------------
# IDENTIFY FLAW
# --------------------------------------------------

def identify_flaw(features):

    rate = features["speech_rate_change"]

    duration = features["duration_change"]

    if rate >= 30 and duration <= -20:
        return "EXCESSIVE SPEECH RATE"

    elif rate <= -30:
        return "SLOW SPEECH RATE"

    elif features["energy_change"] <= -30:
        return "LOW VOCAL ENERGY"

    elif features["pitch_variation_change"] <= -20:
        return "LOW PITCH VARIATION"

    else:
        return "DELIVERY VARIATION"


# --------------------------------------------------
# CAUSAL EXPLANATION
# --------------------------------------------------

def create_explanation(features, flaw):

    rate = features["speech_rate_change"]

    duration = features["duration_change"]

    energy = features["energy_change"]

    variation = features["pitch_variation_change"]

    if flaw == "EXCESSIVE SPEECH RATE":

        explanation = (
            f"The speaker delivered the affected speech region "
            f"{rate:.2f}% faster than the ideal reference. "
            f"The delivery duration decreased by "
            f"{abs(duration):.2f}%, indicating compressed speech. "
        )

        if energy <= -20:

            explanation += (
                f"Vocal energy also decreased by "
                f"{abs(energy):.2f}%, suggesting reduced vocal "
                f"strength during the fast delivery. "
            )

        if variation <= -10:

            explanation += (
                f"Pitch variation decreased by "
                f"{abs(variation):.2f}%, indicating reduced "
                f"expressiveness. "
            )

        explanation += (
            "Together, these measurements provide acoustic "
            "evidence for excessive speaking speed."
        )

        return explanation

    elif flaw == "SLOW SPEECH RATE":

        return (
            f"The speaker's speech rate decreased by "
            f"{abs(rate):.2f}% compared with the ideal reference. "
            f"The delivery duration increased, indicating slower "
            f"speech and possible excessive pauses."
        )

    elif flaw == "LOW VOCAL ENERGY":

        return (
            f"Vocal energy decreased by "
            f"{abs(energy):.2f}% compared with the ideal reference. "
            "This indicates a weaker vocal delivery."
        )

    elif flaw == "LOW PITCH VARIATION":

        return (
            f"Pitch variation decreased by "
            f"{abs(variation):.2f}%. "
            "This suggests a flatter and less expressive delivery."
        )

    return (
        "The acoustic characteristics of the flawed speech "
        "differ from the ideal reference."
    )


# --------------------------------------------------
# RECOMMENDATION
# --------------------------------------------------

def create_recommendation(flaw):

    if flaw == "EXCESSIVE SPEECH RATE":

        return (
            "Slow down during the highlighted region. "
            "Introduce short pauses between major phrases "
            "and clearly pronounce important words."
        )

    elif flaw == "SLOW SPEECH RATE":

        return (
            "Maintain a slightly faster and more consistent "
            "speaking pace. Avoid unnecessary pauses."
        )

    elif flaw == "LOW VOCAL ENERGY":

        return (
            "Increase vocal strength and maintain a clearer "
            "and more audible delivery."
        )

    elif flaw == "LOW PITCH VARIATION":

        return (
            "Use more natural pitch variation to make the "
            "delivery more expressive and engaging."
        )

    return (
        "Compare the highlighted region with the ideal "
        "reference and improve delivery consistency."
    )


# --------------------------------------------------
# BUILD FINAL RESULT
# --------------------------------------------------

def analyze(features):

    flaw = identify_flaw(features)

    score = calculate_score(features)

    severity = get_severity(score)

    confidence = get_confidence(features)

    explanation = create_explanation(
        features,
        flaw
    )

    recommendation = create_recommendation(
        flaw
    )

    result = {

        "flaw_type": flaw,

        "severity": severity,

        "causal_score": round(
            score,
            2
        ),

        "confidence": round(
            confidence,
            2
        ),

        "evidence": {

            "speech_rate_change": round(
                features["speech_rate_change"],
                2
            ),

            "duration_change": round(
                features["duration_change"],
                2
            ),

            "pitch_change": round(
                features["pitch_change"],
                2
            ),

            "energy_change": round(
                features["energy_change"],
                2
            ),

            "pitch_variation_change": round(
                features["pitch_variation_change"],
                2
            ),

            "mfcc_distance": round(
                features["mfcc_distance"],
                2
            )
        },

        "causal_explanation": explanation,

        "recommendation": recommendation
    }

    return result


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print()
    print("======================================")
    print("CAUSAL FLAW SCORING")
    print("======================================")

    result = analyze(
        FEATURES
    )

    print()

    print(
        "Flaw Type:",
        result["flaw_type"]
    )

    print(
        "Severity:",
        result["severity"]
    )

    print(
        "Causal Score:",
        result["causal_score"],
        "/ 100"
    )

    print(
        "Confidence:",
        result["confidence"]
    )

    print()
    print("--------------------------------------")
    print("EVIDENCE")
    print("--------------------------------------")

    for key, value in result["evidence"].items():

        print(
            f"{key}: {value}"
        )

    print()
    print("--------------------------------------")
    print("CAUSAL EXPLANATION")
    print("--------------------------------------")

    print(
        result["causal_explanation"]
    )

    print()
    print("--------------------------------------")
    print("RECOMMENDATION")
    print("--------------------------------------")

    print(
        result["recommendation"]
    )

    print()
    print("======================================")

    # Save JSON
    with open(
        "causal_result.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print()
    print(
        "Result saved to causal_result.json"
    )


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":
    main()
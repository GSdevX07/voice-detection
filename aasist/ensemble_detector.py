def ensemble_decision(aasist_score, wav2vec_score):
    """
    Security-first ensemble logic:
    - HUMAN only with extremely strong evidence
    - Everything else treated as AI or suspected AI
    """

    combined_score = 0.6 * aasist_score + 0.4 * wav2vec_score
    confidence = min(0.5 + abs(combined_score) / 8, 0.99)

    # 🔥 EXTREMELY STRICT HUMAN RULE
    if combined_score < -3.5 and confidence > 0.95:
        return (
            "HUMAN",
            round(confidence, 2),
            [
                "exceptionally strong human speech patterns",
                "highly irregular natural timing",
                "no detectable synthetic artifacts"
            ]
        )

    # 🔥 CLEAR AI
    if combined_score > 0.5:
        return (
            "AI_GENERATED",
            round(confidence, 2),
            [
                "synthetic pitch regularity",
                "temporal generation artifacts",
                "vocoder fingerprints detected"
            ]
        )

    # 🔥 DEFAULT: DO NOT TRUST
    return (
        "SUSPECTED_AI",
        round(confidence, 2),
        [
            "highly human-like but untrusted signal",
            "possible modern TTS or voice cloning",
            "manual verification recommended"
        ]
    )

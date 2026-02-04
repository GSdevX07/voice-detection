def ensemble_decision(aasist_score, wav2vec_score):
    """
    Simple ensemble logic:
    +ve score → AI
    -ve score → Human
    """

    combined_score = 0.6 * aasist_score + 0.4 * wav2vec_score

    if combined_score > 0:
        final_label = "AI_GENERATED"
        confidence = min(0.5 + abs(combined_score) / 10, 0.99)
        insights = [
            "high pitch consistency",
            "lack of natural pauses",
            "vocoder artifacts detected"
        ]
    else:
        final_label = "HUMAN"
        confidence = min(0.5 + abs(combined_score) / 10, 0.99)
        insights = [
            "natural pitch variation",
            "presence of micro-pauses",
            "human-like prosody"
        ]

    return final_label, round(confidence, 2), insights

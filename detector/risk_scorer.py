class RiskScorer:

    def calculate(
        self,
        text_score,
        semantic_score,
        image_score=0
    ):

        # -----------------------------
        # Weighted scoring
        # -----------------------------

        final_score = (
            (text_score * 0.3) +
            (semantic_score * 0.6) +
            (image_score * 0.1)
        )

        # -----------------------------
        # Risk levels
        # -----------------------------

        if final_score >= 0.75:

            level = "HIGH RISK"

        elif final_score >= 0.45:

            level = "MEDIUM RISK"

        else:

            level = "LOW RISK"

        return {
            "risk_score": round(final_score, 2),
            "risk_level": level
        }
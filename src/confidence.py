class ConfidenceScorer:
    def compute(self, docs, answer, verifier):
        scores = []

        for doc in docs:
            score = verifier.check_entailment(doc, answer)
            scores.append(score)

        if len(scores) == 0:
            return 0.0

        # Use max instead of avg (more stable)
        return max(scores)
class ConfidenceScorer:
    def compute(self, docs, answer, verifier):
        scores = []

        for doc in docs:
            score = verifier.check_entailment(doc, answer)
            scores.append(score)

        if len(scores) == 0:
            return 0.0

        return sum(scores) / len(scores)
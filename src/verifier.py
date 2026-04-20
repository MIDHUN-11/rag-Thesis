from transformers import pipeline

class Verifier:
    def __init__(self):
        self.nli = pipeline(
            "text-classification",
            model="MoritzLaurer/deberta-v3-base-mnli-fever-anli"
        )

    def check_entailment(self, doc, answer):
        result = self.nli({
            "text": doc,
            "text_pair": answer
        })

        # 🔥 Handle both dict and list outputs
        if isinstance(result, list):
            r = result[0]
        else:
            r = result

        label = r["label"]
        score = r["score"]

        if label in ["ENTAILMENT", "LABEL_2"]:
            return score
        elif label in ["NEUTRAL", "LABEL_1"]:
            return score * 0.3
        else:  # contradiction
            return -score
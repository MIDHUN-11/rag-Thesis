from transformers import pipeline

class Verifier:
    def __init__(self):
        self.nli = pipeline(
            "text-classification",
            model="MoritzLaurer/deberta-v3-base-mnli-fever-anli"
        )

    def check_entailment(self, doc, answer):
        result = self.nli(f"{doc} </s></s> {answer}")

        r = result[0]
        label = r["label"]
        score = r["score"]

        if label in ["ENTAILMENT", "LABEL_2"]:
            return score
        else:
            return 0.0
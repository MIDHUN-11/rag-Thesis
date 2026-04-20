from src.retrieval import Retriever
from src.generator import Generator
from src.verifier import Verifier
from src.confidence import ConfidenceScorer

class RAGPipeline:
    def __init__(self, docs):
        self.retriever = Retriever(docs)
        self.generator = Generator()
        self.verifier = Verifier()
        self.confidence = ConfidenceScorer()

    def normalize_answer(self, answer, query):
        # If answer is too short, expand it for NLI
        if len(answer.split()) <= 3:
            return f"{answer} is the answer to: {query}"
        return answer

    def run(self, query):
        docs = self.retriever.search(query)
        context = "\n".join(docs)

        answer = self.generator.generate(context, query).strip()

        # 🔥 normalize for better NLI
        normalized_answer = self.normalize_answer(answer, query)

        confidence = self.confidence.compute(
            docs,
            normalized_answer,
            self.verifier
        )

        return {
            "query": query,
            "answer": answer,
            "evidence": docs,
            "confidence": round(confidence, 2)
        }
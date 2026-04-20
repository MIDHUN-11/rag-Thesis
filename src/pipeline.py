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

    def run(self, query):
        docs = self.retriever.search(query)
        context = "\n".join(docs)

        answer = self.generator.generate(context, query)
        confidence = self.confidence.compute(docs, answer, self.verifier)

        return {
            "query": query,
            "answer": answer.strip(),
            "evidence": docs,
            "confidence": round(confidence, 2)
        }
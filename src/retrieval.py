from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        embeddings = self.model.encode(docs)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def search(self, query, top_k=1):  # 🔥 reduced from 2 → 1
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb), top_k)
        return [self.docs[i] for i in I[0]]
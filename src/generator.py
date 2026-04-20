import requests

class Generator:
    def __init__(self, model="llama3"):
        self.model = model

    def generate(self, context, query):
        prompt = f"""
Use ONLY the context below.
If answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]
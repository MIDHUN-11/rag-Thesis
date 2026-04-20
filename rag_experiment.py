from src.pipeline import RAGPipeline

def load_docs():
    with open("data/docs.txt") as f:
        return [line.strip() for line in f.readlines()]

def main():
    docs = load_docs()
    rag = RAGPipeline(docs)

    queries = [
        "Who is CEO of OpenAI?",
        "Who is CEO of Microsoft?",
        "Who founded SpaceX?",
        "Who founded OpenAI?"
    ]

    for q in queries:
        result = rag.run(q)

        print("\n==============================")
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Evidence: {result['evidence']}")
        print(f"Confidence: {result['confidence']}")

if __name__ == "__main__":
    main()
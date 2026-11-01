from dotenv import load_dotenv
load_dotenv()

from agentic_rag.graph.graph import app

if __name__ == "__main__":
    result = app.invoke({"question": "What is agent memory?"})

    print(f"Question: {result['question']}")
    print(f"\nAnswer:\n{result['generation']}")

    sources = {
        source
        for document in result.get("documents", [])
        if (source := document.metadata.get("source") or document.metadata.get("url"))
    }
    if sources:
        print("\nSources:")
        for source in sorted(sources):
            print(f"- {source}")

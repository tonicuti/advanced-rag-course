from typing import Dict, Any

from agentic_rag.graph.state import GraphState
from agentic_rag.graph.chains.generation import generation_chain

def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generates a response based on the question and relevant documents in the graph state.

    Args:
        state: The current state of the graph.
    """
    
    print("--- GENERATE ---")
    question = state["question"]
    documents = state["documents"]
    
    res = generation_chain.invoke(
        {
            "question": question,
            "context": "\n\n".join(doc.page_content for doc in documents),
        }
    )
    
    return {"documents": documents, "question": question, "generation": res}

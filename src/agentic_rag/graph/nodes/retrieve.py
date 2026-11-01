from typing import Any, Dict 

from ..state import GraphState
from ...ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves relevant documents based on the question in the graph state.

    Args:
        state: The current state of the graph.
    """
    
    print("--- RETRIEVE ---")
    question = state["question"]
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question": question}


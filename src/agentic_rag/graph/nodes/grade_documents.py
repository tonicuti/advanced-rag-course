from typing import Any, Dict

from agentic_rag.graph.state import GraphState
from agentic_rag.graph.chains.retrieval_grader import retrieval_grader

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state(dict): The current state of the graph.
    
    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    
    print("--- CHECK DOCUMENTS RELEVANT TO THE QUESTION ---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"graded_documents": [], "question": question}
    
    filtered_docs = []
    web_search = False
    
    for doc in documents:
        doc_txt = doc.page_content
        score = retrieval_grader.invoke({"question": question, "document": doc_txt})
        
        grade = score.binary_score
        if grade.lower() == "yes":
            print("--- GRADE: DOCUMENT IS RELEVANT ---")
            filtered_docs.append(doc)
        else:
            print("--- GRADE: DOCUMENT IS NOT RELEVANT ---")
            web_search = True
            continue
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
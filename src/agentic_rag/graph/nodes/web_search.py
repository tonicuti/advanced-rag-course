from typing import Dict, Any

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from agentic_rag.graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max_results = 3)

def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Performs a web search based on the question in the graph state.

    Args:
        state: The current state of the graph.
    """
    
    print("--- WEB SEARCH ---")
    question = state["question"]
    documents = list(state.get("documents") or [])
    
    tavily_response = web_search_tool.invoke({"query": question})
    if "error" in tavily_response:
        raise RuntimeError("Tavily search failed") from tavily_response["error"]

    joined_tavily_results = "\n".join(
        result["content"] for result in tavily_response["results"]
    )
    
    web_search_results = Document(
        page_content = joined_tavily_results
    )
    
    documents.append(web_search_results)
    
    return {"documents": documents, "question": question}

if __name__ == "__main__":
    # Example usage 
    web_search(state = {"question": "What is agent memory?", "documents": None})

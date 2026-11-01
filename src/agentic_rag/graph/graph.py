from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, StateGraph
from agentic_rag.graph.consts import RETRIEVE, GRADE_DOCUMENTS, WEB_SEARCH, GENERATE
from agentic_rag.graph.nodes import retrieve, grade_documents, web_search, generate
from agentic_rag.graph.state import GraphState
from agentic_rag.graph.chains.hallucination_grader import hallucination_grader, GradeHallucination
from agentic_rag.graph.chains.answer_grader import answer_grader, GradeAnswer
from agentic_rag.graph.chains.router import RouteQuery, question_router

def decide_to_generate(state: GraphState) -> bool:
    """
    Determines whether to generate a response based on the current state.

    Args:
        state: The current state of the graph.

    Returns:
        A boolean indicating whether to generate a response.
    """
    
    print("--- ASSESS GRADE DOCUMENTS ---")
    
    if state["web_search"]:
        print("--DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH RESULTS--")
        return WEB_SEARCH
    else:
        print("-- DECISION: GENERATE ")
        return GENERATE
    
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = "\n\n".join(
        document.page_content for document in state.get("documents") or []
    )
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if score.binary_score.strip().lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if score.binary_score.strip().lower() == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEB_SEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEB_SEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE
    
workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(WEB_SEARCH, web_search)
workflow.add_node(GENERATE, generate)

workflow.set_conditional_entry_point(
    route_question,
    path_map = {
        WEB_SEARCH: WEB_SEARCH,
        RETRIEVE: RETRIEVE,
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS, 
    decide_to_generate,
    path_map={
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE
    }
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEB_SEARCH,
    },
)
workflow.add_edge(WEB_SEARCH, GENERATE)

app = workflow.compile()

# app.get_graph().draw_mermaid_png(output_file_path = "graph.png")

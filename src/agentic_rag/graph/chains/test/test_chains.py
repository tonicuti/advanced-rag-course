from dotenv import load_dotenv
from pprint import pprint

load_dotenv()  

from agentic_rag.graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from agentic_rag.ingestion import retriever
from agentic_rag.graph.chains.generation import generation_chain
from agentic_rag.graph.chains.hallucination_grader import hallucination_grader, GradeHallucination
from agentic_rag.graph.chains.router import RouteQuery, question_router

# def test_retrieval_grader():
#     question = "What is the agent memory?"
    
#     docs = retriever.invoke(question)

#     assert docs, "The retriever returned no documents for the test question."
#     doc_txt = docs[0].page_content
    
#     res: GradeDocuments = retrieval_grader.invoke(
#         {"question": question, "document": doc_txt}
#     )
    
#     print("The content of the Document: ", doc_txt)
    
#     assert res.binary_score == "Yes"

# def test_generation_chain() -> None:
#     question = "What is the agent memory?"
    
#     docs = retriever.invoke(question)

#     assert docs, "The retriever returned no documents for the test question."
    
#     res = generation_chain.invoke(
#         {
#             "question": question,
#             "context": "\n\n".join(doc.page_content for doc in docs),
#         }
#     )
    
#     pprint(res)
    
    
# def test_hallucination_grader() -> None:
#     question = "What is the agent memory?"
    
#     docs = retriever.invoke(question)

#     assert docs, "The retriever returned no documents for the test question."
    
#     generation = generation_chain.invoke(
#         {
#             "question": question,
#             "context": "\n\n".join(doc.page_content for doc in docs),
#         }
#     )
    
#     res: GradeHallucination = hallucination_grader.invoke(
#         {
#             "documents": "\n\n".join(doc.page_content for doc in docs),
#             "generation": generation,
#         }
#     )
    
#     assert res.binary_score

def test_router() -> None:
    question = "What is the agent memory?"
    
    res: RouteQuery = question_router.invoke({"question": question})
    
    assert res.datasource == "vectorstore"
    
def test_router_websearch() -> None:
    question = "What is the weather in New York City today?"
    
    res: RouteQuery = question_router.invoke({"question": question})
    
    assert res.datasource == "websearch"
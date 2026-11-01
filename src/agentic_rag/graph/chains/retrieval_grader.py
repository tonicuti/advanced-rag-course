from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "qwen3:1.7b",
    temperature = 0
)

class GradeDocuments(BaseModel):
    """Binary score for relevant check on retrieved documents."""
    
    binary_score: str = Field(
        description = "Documents are relevant to the question or not. 'Yes' or 'No'."
    )
    
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """ 
    You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains the keyword(s) or semantic meaning related to the question, please grade it as relevant. \n
    Give a binary score "Yes" or "No" score to indicate whether the document is relevant to the question.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document}\n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
        
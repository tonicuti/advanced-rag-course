from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "qwen3:1.7b",
    temperature = 0
)

class GradeHallucination(BaseModel):
    """Binary score for hallucination detection."""
    
    binary_score: str = Field(
        description = "Answer is grounded in the fact, 'yes' or 'no'."
    )
    
structured_llm_grader = llm.with_structured_output(GradeHallucination)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
     
     
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n{documents} \n\nLLM Generation: {generation}"),
    ]
)

hallucination_grader = RunnableSequence = hallucination_prompt | structured_llm_grader
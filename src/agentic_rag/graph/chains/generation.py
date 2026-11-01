from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "qwen3:1.7b",
    temperature = 0
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """ 
            You are an assistant for question answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you do not know the answer, just say that you don't know. Do not try to make up an answer.
            Use THREE sentences maximum and keep the answer concise.
            Question: {question} \nContext: {context} \nAnswer: 
            """
        ),
    ]
)


generation_chain = prompt | llm | StrOutputParser()

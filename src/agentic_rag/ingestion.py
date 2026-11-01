from dotenv import load_dotenv
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

CHROMA_DIR = str(Path(__file__).resolve().parent / ".chroma")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

docs = [UnstructuredLoader(web_url = url, chunking_strategy = "basic", max_characters = 1000000).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size = 250, 
    chunk_overlap = 0
)

docs_split = text_splitter.split_documents(docs_list)

# vectorstore = Chroma.from_documents(
#     documents = docs_split,
#     collection_name = "advanced-rag-chroma",
#     embedding = OllamaEmbeddings(model = "embeddinggemma:latest"),
#     persist_directory = CHROMA_DIR,
# )

retriever = Chroma(
    collection_name = "advanced-rag-chroma",
    embedding_function = OllamaEmbeddings(model = "embeddinggemma:latest"),
    persist_directory = CHROMA_DIR,
).as_retriever()

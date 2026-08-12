from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma         #chromadb installaion

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=api_key
)
loader = PyPDFLoader("document loaders/conference.pdf")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

texts = [chunk.page_content for chunk in chunks]


embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

embeddings = embedding_model.embed_documents(texts)

print(f"Generated {len(embeddings)} embeddings")
print(f"Dimensions of each embedding: {len(embeddings[0])}")

print(embeddings[0][:10])

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("\nVector Database Created Successfully!")
print("Database saved in './chroma_db'")
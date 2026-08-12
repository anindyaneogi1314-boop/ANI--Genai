from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

# Load API Key
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# Initialize Mistral LLM
llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=api_key
)
loader = PyPDFLoader("document loaders/cse.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50
)

split = splitter.split_documents(docs)
document_text = split

print(len(split))
# Ask user question
question = input("Ask a question: ")

# Create prompt
prompt = f"""
Use the following document to answer the question.

Document:
{document_text}

Question:
{question}

Answer:
"""

# Generate response
response = llm.invoke([HumanMessage(content=prompt)])

print("\nAnswer:\n")
print(response.content)
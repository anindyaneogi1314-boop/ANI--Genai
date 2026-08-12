import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

CHROMA_PATH = "./chroma_db"

# ==========================================
# 1. Load the existing Chroma database
# ==========================================
# We use the exact same embedding model configuration so it matches your build script
embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=api_key
)

db = Chroma(
    persist_directory=CHROMA_PATH, 
    embedding_function=embedding_model
)

# ==========================================
# 2 & 3. Embed query & Retrieve using MMR
# ==========================================
# MMR filters out text chunks that look too similar to each other,
# which is incredibly helpful when dealing with mixed files/sites.
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 5}
)

question = input("Ask your question: ")
relevant_docs = retriever.invoke(question)

# --- Display All Retrieved Chunks ---
print("\n" + "=" * 60)
print(" RETRIEVED CHUNKS (via MMR)")
print("=" * 60)

for i, doc in enumerate(relevant_docs, start=1):
    # This automatically tracks whether it came from a URL, PDF, or .txt file!
    source = doc.metadata.get("source", "Unknown Source")
    print(f"\n[Chunk {i}] | Source: {source}")
    print("-" * 40)
    print(doc.page_content)
    print("-" * 40)

# ==========================================
# 4. Give ONLY those chunks to the LLM
# ==========================================
context = "\n\n".join(
    doc.page_content
    for doc in relevant_docs
)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
formatted_prompt = template.format(context=context, question=question)

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=api_key,
    temperature=0
)

response = llm.invoke(formatted_prompt)

# ==========================================
# 5. Print the final answer
# ==========================================
print("\n" + "=" * 60)
print(" FINAL ANSWER")
print("=" * 60)
print(response.content)
print("=" * 60)
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

# This list will hold all documents extracted from different sources
all_docs = []
source_dir = "document loaders"

# =====================================================================
# 1. Load All Different Data Sources
# =====================================================================

print("Scanning and loading documents...")

# --- A. Load PDFs ---
pdf_path = os.path.join(source_dir, "cse.pdf")
if os.path.exists(pdf_path):
    print(f" -> Loading PDF: {pdf_path}")
    pdf_loader = PyPDFLoader(pdf_path)
    all_docs.extend(pdf_loader.load())

# --- B. Load Local Text Files (*.txt) ---
if os.path.exists(source_dir):
    print(f" -> Scanning for text files in: '{source_dir}/'")
    # Automatically finds and loads every .txt file in the folder
    dir_loader = DirectoryLoader(source_dir, glob="*.txt", loader_cls=TextLoader)
    all_docs.extend(dir_loader.load())

# --- C. Load Websites (URLs) ---
urls = [
    "https://nit.ac.in/",
    
]
print(f" -> Scraping websites: {len(urls)} URLs targeted")
web_loader = WebBaseLoader(urls)
all_docs.extend(web_loader.load())

# --- D. Load Raw Text Strings (Manual Input) ---
raw_text = "This is a custom text string about an emergency backup protocol or specific note."
print(" -> Injecting manual text strings")
text_document = Document(
    page_content=raw_text, 
    metadata={"source": "manual_input"}
)
all_docs.append(text_document)


# =====================================================================
# 2. Split Everything Jointly Into Chunks
# =====================================================================
print(f"\nSplitting a total of {len(all_docs)} loaded documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_docs)
print(f"Generated {len(chunks)} text chunks.")


# =====================================================================
# 3. Generate Embeddings & Create/Update Vector Store
# =====================================================================
print("\nInitializing Mistral embedding model...")
embedding_model = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=api_key
)

print("Writing chunks to Chroma Database at './chroma_db'...")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("\n" + "="*40)
print(" Vector Database Built Successfully!")
print(f" Total chunks now stored: {len(chunks)}")
print("="*40)
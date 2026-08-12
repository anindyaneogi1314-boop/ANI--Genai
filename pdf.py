from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader("document loaders/cse.pdf")  

docs = loader.load()
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(docs)


print(docs[0])
print(len(docs))

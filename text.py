from langchain_community.document_loaders import TextLoader
loader = TextLoader("document loaders/notes.txt")  

docs = loader.load()

print(docs[0])
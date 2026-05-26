import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the PDF from your data folder
print("--- Loading PDF ---")
loader = PyPDFLoader("data/company_info.pdf") # Make sure this matches your file name!
data = loader.load()

# 2. Split the text into smaller chunks (so the AI can find specific info)
print("--- Splitting Text into Chunks ---")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# 3. Create the "Embeddings" (The FREE HuggingFace model)
print("--- Creating Embeddings (This may take a minute the first time) ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Save to our local ChromaDB database
print("--- Saving to Database ---")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("Finished! You now have a searchable 'Brain' in the chroma_db folder.")
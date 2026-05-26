from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the same embedding model
print("--- Loading Embeddings ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to the database you just built
print("--- Connecting to Database ---")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3. Ask a question! 
question = "What is your data backup procedure?" 

# 4. Search the database for the best matching chunks of text
print(f"\nSearching for: '{question}'\n")
results = vector_db.similarity_search(question, k=2)

# 5. Print the results
for i, result in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(result.page_content)
    print("\n")
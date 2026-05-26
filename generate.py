import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Load your secret API key
load_dotenv()

# 2. Connect to Database (Just like before)
print("--- Connecting to Database ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3. Connect to the Free Groq AI (Llama 3)
print("--- Waking up the AI ---")
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant",
    api_key="gsk_your_key_here"
)

# 4. Set up the strict AI Instructions (The Prompt)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert enterprise sales writer answering an RFP. Answer the question using ONLY the provided context. Do not make anything up. If the answer is not in the context, say 'Information not provided.'"),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

# 5. The Question we want to ask
question = "How quickly can you recover data if the system crashes?"

# 6. Retrieve the relevant data from the database
results = vector_db.similarity_search(question, k=2)

# Combine the results into a single string of text
context_text = "\n\n".join([doc.page_content for doc in results])

# 7. Ask the AI to write the final answer
print(f"\nQuestion: {question}")
print("Generating Answer...\n")

# Format the prompt and run it through the LLM
formatted_prompt = prompt_template.format_messages(context=context_text, question=question)
response = llm.invoke(formatted_prompt)

print("--- AI RFP RESPONSE ---")
print(response.content)
print("\n-----------------------")
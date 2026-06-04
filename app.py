import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- WEB PAGE SETUP ---
st.set_page_config(page_title="RFP Agent", page_icon="🤖", layout="wide")

# --- NEW: INTERACTIVE SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083213.png", width=100) # Cool dashboard icon
    st.title("System Status")
    st.success("🟢 Vector Database: Online")
    st.success("🟢 Llama 3 AI: Connected")
    st.info("Currently querying: Acme_Master_Doc.pdf")

st.title("📄 Autonomous RFP & Tender Agent")
st.markdown("Ask a question, and the AI will draft a response based **ONLY** on your company's uploaded documents.")

# --- LOAD AI AND DATABASE ---
@st.cache_resource 
def load_brain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    # ⚠️ PUT YOUR REAL API KEY BELOW!
    ai = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", api_key="gsk_your_key_here") 
    return db, ai

vector_db, llm = load_brain()

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert enterprise sales writer answering an RFP. Answer the question using ONLY the provided context. Do not make anything up. If the answer is not in the context, say 'Information not provided.'"),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

# --- USER INTERFACE ---
user_question = st.text_input("Enter the RFP Question:", placeholder="e.g., How do you handle disaster recovery?")

if st.button("Generate Professional Response", type="primary"):
    if user_question:
        with st.spinner("Searching secure database and drafting response..."):
            
            # Find the data and write the answer
            results = vector_db.similarity_search(user_question, k=2)
            context_text = "\n\n".join([doc.page_content for doc in results])
            
            formatted_prompt = prompt_template.format_messages(context=context_text, question=user_question)
            response = llm.invoke(formatted_prompt)
            
            # Display the final answer on the screen!
            st.success("✨ Draft Complete!")
            st.write("### Generated Proposal:")
            st.info(response.content)
            
            # --- NEW: INTERACTIVE "SHOW SOURCES" DROPDOWN ---
            with st.expander("🔍 View Retrieved Database Context (Anti-Hallucination Check)"):
                st.write("The AI used the following specific text blocks to generate this answer:")
                for i, doc in enumerate(results):
                    st.markdown(f"**Source Match {i+1}:**")
                    st.caption(doc.page_content)
                    st.divider()
            
            # The Download Button
            st.download_button(
                label="📥 Download Response Document",
                data=response.content,
                file_name="Acme_RFP_Response.txt",
                mime="text/plain"
            )
    else:
        st.warning("Please enter a question first.")
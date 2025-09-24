# app.py -- Streamlit front‑end for Cyber SOC SOP Assistant
import os
import streamlit as st
import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from sop_loader import build_vector_store, collection_count

# ---------------------------
# Configuration -- MUST match sop_loader.py
# ---------------------------
SOP_FILE = "sops/sop.pdf"            # Path to your SOP file
DB_PATH = "sop_db"                   # Persistent DB dir -- same in both files
COLLECTION_NAME = "sop_chunks"       # Must match in build_vector_store/query
MODEL_PATH = "mistral.gguf"          # Path to local LLaMA/Mistral model

# Optional: import your local LLaMA wrapper if you have one
from model_interface import LlamaSOPAssistant

# ---------------------------
# Streamlit title
# ---------------------------
st.title("Cyber SOC SOP Assistant")

# ---------------------------
# Persistent Chroma client (Streamlit cache to survive reruns)
# ---------------------------
@st.cache_resource
def get_client():
   os.makedirs(DB_PATH, exist_ok=True)
   return chromadb.PersistentClient(path=DB_PATH)

client = get_client()

# ---------------------------
# Indexing workflow
# ---------------------------
if st.button("Index SOPs"):
   st.info("Rebuilding SOP index...")
   result = build_vector_store(
       sop_file=SOP_FILE,
       db_path=DB_PATH,
       collection_name=COLLECTION_NAME,
       reset=True
   )
   st.success(f"SOP index rebuilt. Indexed {result['chunks_indexed']} chunks.")

# ---------------------------
# Query workflow
# ---------------------------
query = st.text_area("Enter your question:")

if st.button("Get Answer"):
   if not query.strip():
       st.warning("Please enter a question.")
   else:
       # Check if collection exists
       try:
           collection = client.get_collection(COLLECTION_NAME)
       except NotFoundError:
           st.error("No SOP index found. Click 'Index SOPs' first.")
           st.stop()

       # Check if collection has data
       if collection_count(DB_PATH, COLLECTION_NAME) == 0:
           st.warning("No SOP chunks found. Click 'Index SOPs' to build the database.")
           st.stop()

       # Embed query
       embedder = SentenceTransformer("all-MiniLM-L6-v2")
       query_embedding = embedder.encode(query).tolist()

       # Retrieve top results
       results = collection.query(
           query_embeddings=[query_embedding],
           n_results=3
       )
       relevant_chunks = results.get("documents", [[]])[0]

       if not relevant_chunks:
           st.info("No relevant chunks found. Try rephrasing or re‑indexing the SOPs.")
       else:
           sop_context = "\n\n".join(relevant_chunks)
           prompt = f"""You are a SOC assistant. Here are the relevant sections from the SOP:
{sop_context}

User question: {query}
Answer based only on the provided SOP content."""

           # Call local LLaMA/Mistral assistant
           assistant = LlamaSOPAssistant(model_path=MODEL_PATH)
           answer = assistant.query(prompt)

           st.subheader("Answer:")
           st.write(answer)

# ---------------------------
# Debug info (optional)
# ---------------------------
with st.expander("Debug info"):
   st.write("Collections:", [c.name for c in client.list_collections()])
   st.write("Chunk count:", collection_count(DB_PATH, COLLECTION_NAME))

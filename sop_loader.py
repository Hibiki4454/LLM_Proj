from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

# Optional dependency for PDFs
try:
   import pdfplumber
   _PDF_OK = True
except Exception:
   _PDF_OK = False


# ----------------------------
# File loading
# ----------------------------
def load_text(file_path: str) -> str:
   """
   Load SOP text from a PDF or TXT file.
   Returns a single normalized string.
   """
   if not os.path.exists(file_path):
       raise FileNotFoundError(f"SOP file not found: {file_path}")

   ext = os.path.splitext(file_path)[1].lower()

   if ext == ".txt":
       with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
           text = f.read()
       return normalize_text(text)

   if ext == ".pdf":
       if not _PDF_OK:
           raise RuntimeError("pdfplumber is not installed but a PDF was provided.")
       pages: List[str] = []
       with pdfplumber.open(file_path) as pdf:
           for p in pdf.pages:
               t = p.extract_text() or ""
               pages.append(t)
       return normalize_text("\n\n".join(pages))

   raise ValueError(f"Unsupported file extension: {ext}. Use .pdf or .txt")


def normalize_text(text: str) -> str:
   """
   Light normalization to reduce noisy whitespace.
   """
   # Collapse repeated whitespace but keep paragraphs
   lines = [ln.strip() for ln in text.splitlines()]
   cleaned = "\n".join(ln for ln in lines if ln != "")
   return cleaned.strip()


# ----------------------------
# Chunking
# ----------------------------
def chunk_text(
   text: str,
   chunk_size: int = 800,
   overlap: int = 150,
) -> Tuple[List[str], List[Tuple[int, int]]]:
   """
   Split text into overlapping character chunks.
   Returns:
     - chunks: list of text chunks
     - spans:  list of (start_idx, end_idx) for each chunk in original text
   """
   if not text:
       return [], []

   if overlap >= chunk_size:
       raise ValueError("overlap must be smaller than chunk_size")

   chunks: List[str] = []
   spans: List[Tuple[int, int]] = []

   n = len(text)
   start = 0
   while start < n:
       end = min(start + chunk_size, n)
       chunk = text[start:end]

       # Try to avoid mid-word endings by backing up to last whitespace
       if end < n:
           back = chunk.rfind(" ")
           if back > 0 and (end - (start + back)) < 60:  # only if reasonably close
               end = start + back
               chunk = text[start:end]

       chunks.append(chunk)
       spans.append((start, end))

       if end == n:
           break

       start = max(0, end - overlap)

   return chunks, spans


# ----------------------------
# Chroma helpers
# ----------------------------
def get_persistent_client(db_path: str) -> chromadb.api.types.ClientAPI:
   """
   Create or connect to a persistent ChromaDB at the given path.
   """
   os.makedirs(db_path, exist_ok=True)
   return chromadb.PersistentClient(path=db_path)


def get_or_create_collection(
   client: chromadb.api.types.ClientAPI,
   collection_name: str,
):
   """
   Get collection if it exists, otherwise create it (cosine space for embeddings).
   """
   try:
       return client.get_collection(collection_name)
   except NotFoundError:
       return client.create_collection(
           name=collection_name,
           metadata={"hnsw:space": "cosine"},
       )


def reset_collection(
   client: chromadb.api.types.ClientAPI,
   collection_name: str,
):
   """
   Drop collection if it exists, then recreate it.
   """
   try:
       client.delete_collection(collection_name)
   except NotFoundError:
       pass
   return client.create_collection(
       name=collection_name,
       metadata={"hnsw:space": "cosine"},
   )


# ----------------------------
# Embeddings
# ----------------------------
def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
   """
   Load a SentenceTransformer embedder. Keep model small for air-gapped CPU.
   """
   return SentenceTransformer(model_name)


def embed_chunks(
   embedder: SentenceTransformer,
   chunks: List[str],
   batch_size: int = 64,
) -> List[List[float]]:
   """
   Compute embeddings for chunks in batches, return as list of lists (floats).
   """
   if not chunks:
       return []
   vecs = embedder.encode(chunks, batch_size=batch_size, show_progress_bar=False)
   # Ensure nested Python lists (not np arrays) for Chroma
   return [v.tolist() for v in vecs]


# ----------------------------
# Public API
# ----------------------------
def build_vector_store(
   sop_file: str,
   db_path: str,
   collection_name: str = "sop_chunks",
   reset: bool = True,
   chunk_size: int = 800,
   overlap: int = 150,
   embedding_model: str = "all-MiniLM-L6-v2",
   batch_size: int = 64,
) -> Dict[str, int]:
   """
   Build (or rebuild) a persistent vector store for the given SOP.

   Args:
     sop_file: path to SOP (.pdf or .txt)
     db_path:  directory for Chroma persistence
     collection_name: collection to write into (must match your query side)
     reset: drop + recreate the collection if True
     chunk_size: characters per chunk
     overlap: overlap between adjacent chunks
     embedding_model: sentence-transformers model name
     batch_size: batch size for embedding

   Returns:
     dict with {'chunks_indexed': int}
   """
   # 1) Load + chunk
   text = load_text(sop_file)
   chunks, spans = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

   if not chunks:
       raise ValueError("No text content found to index. Check your SOP file.")

   # 2) Connect persistent DB
   client = get_persistent_client(db_path)

   # 3) Reset or get collection
   collection = reset_collection(client, collection_name) if reset else get_or_create_collection(client, collection_name)

   # 4) Embed
   embedder = get_embedder(embedding_model)
   embeddings = embed_chunks(embedder, chunks, batch_size=batch_size)

   # 5) Prepare metadata + IDs
   ids = [f"{collection_name}_{i:06d}" for i in range(len(chunks))]
   base_meta = {
       "source_file": os.path.abspath(sop_file),
       "collection": collection_name,
   }
   metadatas = []
   for i, (start, end) in enumerate(spans):
       metadatas.append({
           **base_meta,
           "chunk_index": i,
           "char_start": start,
           "char_end": end,
       })

   # 6) Add to Chroma
   collection.add(
       ids=ids,
       documents=chunks,
       metadatas=metadatas,
       embeddings=embeddings,
   )

   return {"chunks_indexed": len(chunks)}


def collection_count(db_path: str, collection_name: str = "sop_chunks") -> int:
   """
   Convenience method: return number of items in the collection.
   """
   client = get_persistent_client(db_path)
   try:
       col = client.get_collection(collection_name)
   except NotFoundError:
       return 0
   return col.count()


def list_collections(db_path: str) -> List[str]:
   """
   List collection names in the given persistent DB path.
   """
   client = get_persistent_client(db_path)
   cols = client.list_collections()
   return [c.name for c in cols]

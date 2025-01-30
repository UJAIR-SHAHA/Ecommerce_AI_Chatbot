import json
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Load product data
product_data = pd.read_csv("model/final_fashion_data_v2.csv")

# Check required columns
if 'product_id' not in product_data.columns or 'tags' not in product_data.columns:
    raise ValueError("CSV must contain 'product_id' and 'tags' columns!")

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for tags
descriptions = product_data['tags'].astype(str).tolist()
embeddings = model.encode(descriptions, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

# Get embedding dimension
dimension = embeddings.shape[1]

# Create FAISS index (Inner Product for cosine similarity)
index = faiss.IndexFlatIP(dimension)

# Normalize embeddings for better similarity search
faiss.normalize_L2(embeddings)

# Add embeddings to FAISS index
index.add(embeddings)

# Ensure index size matches the dataset
assert index.ntotal == len(product_data), "FAISS index size mismatch!"

# Create document store for storing product metadata
docstore = InMemoryDocstore()
docstore.docs = {
    str(i): {"product_id": str(pid), "tags": tags}
    for i, (pid, tags) in enumerate(zip(product_data["product_id"], descriptions))
}

# Define an embedding function for LangChain FAISS
def embedding_function(texts):
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

# Create FAISS vector store
vector_store = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=docstore,
    index_to_docstore_id={i: str(i) for i in range(len(product_data))}
)

# Save FAISS index
vector_store.save_local("faiss_index")

print(f"✅ FAISS index created with {index.ntotal} entries and saved successfully!")

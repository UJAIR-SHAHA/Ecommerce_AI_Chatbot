import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


# Load product data
product_data = pd.read_csv("model/final_fashion_data_v2.csv")

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
descriptions = product_data['tags'].tolist()
embeddings = model.encode(descriptions)

# Convert embeddings to NumPy array and ensure the correct data type
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
# Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.save_local("faiss_index")

import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Load product data
product_data = pd.read_csv("model/final_fashion_data_v2.csv")  # Replace with your file path

def combine_columns(row):
    # Combine all columns into one string
    return " ".join([f"{col}: {str(row[col])}" for col in product_data.columns])

# Create a combined text column
product_data['combined_text'] = product_data.apply(combine_columns, axis=1)

# Extract the combined text for embedding
combined_texts = product_data['combined_text'].tolist()

# Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-l6-v2')
embeddings = model.encode(combined_texts)
embeddings = np.array(embeddings).astype('float32')
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save FAISS index to a file
faiss.write_index(index, "faiss_index.idx")

# Create docstore for relevant data
docstore = {str(i): {"product_id": str(product_data['product_id'][i]), "description": combined_texts[i]} for i in range(len(product_data))}

# Save docstore to a file
with open("docstore.pkl", "wb") as f:
    pickle.dump(docstore, f)

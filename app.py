import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify,render_template
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, pipeline
from dotenv import load_dotenv, find_dotenv
import os
import requests

_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_TOKEN']

# Load the local CSV file containing your product data
product_data = pd.read_csv("model/final_fashion_data_v2.csv")  # Replace with your file path
def combine_columns(row):
    # Combine all columns into one string; adjust as needed based on your columns
    return " ".join([f"{col}: {str(row[col])}" for col in product_data.columns])

# Create a combined text column
product_data['combined_text'] = product_data.apply(combine_columns, axis=1)

# Extract the combined text for embedding
combined_texts = product_data['combined_text'].tolist()

# Extract 'tags' column as descriptions
# Initialize the embedding model (using sentence-transformers/all-MiniLM-l6-v2)
model = SentenceTransformer('sentence-transformers/all-MiniLM-l6-v2')
embeddings = model.encode(combined_texts)

# Convert embeddings to NumPy array and ensure correct data type
embeddings = np.array(embeddings).astype('float32')

# Normalize embeddings for cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Get embedding dimension
dimension = embeddings.shape[1]

# Create a FAISS index (Inner Product for cosine similarity)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Create docstore for relevant data (product_id and tags)
docstore = {str(i): {"product_id": str(product_data['product_id'][i]), "description": combined_texts[i]} for i in
            range(len(product_data))}

# # Initialize tokenizer for the text-generation model
# tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert", padding=True, truncation=True,
#                                           max_length=512)
#
# # Initialize the text-generation pipeline using HuggingFace transformers
# text_generator = pipeline("text-generation", model="Intel/dynamic_tinybert", tokenizer=tokenizer)

# Flask app setup
app = Flask(__name__)


# Define a function to get relevant documents and generate answers
def generate_answer(query):
    # Perform search in the FAISS index for relevant documents
    query_embedding = model.encode([query])[0].astype("float32")
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize query

    # Search the FAISS index
    D, I = index.search(np.array([query_embedding]), k=4)  # Get top 4 relevant documents

    # Get the top result (most relevant document)
    relevant_doc = docstore[str(I[0][0])]
    context = relevant_doc['description']

    system_prompt = f"You are an intelligent eCommerce assistant. Your task is to provide accurate and clear information based on the provided product tags and the user's query. Your responses should focus on the details from the product tags and ensure they are relevant to the query. " \
                    f"### Context (Product Tags): {context} " \
                    f"### User Query: {query} " \
                    f"### Expected Output: "
    # # Generate response using text-generation model
    # generated_response = text_generator(system_prompt, max_length=200, num_return_sequences=1)
    # return generated_response[0]['generated_text']

    # Make an API call to Hugging Face using the model (assuming the model is hosted on Hugging Face)
    url = f"https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    headers = {
        "Authorization": f"Bearer {hf_api_key}"
    }

    payload = {
        "inputs": system_prompt
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        generated_text = result[0]['generated_text'].strip()
        cleaned_response = generated_text.split('### Expected Output:')[-1].strip()
        return cleaned_response
    else:
        return f"Error: {response.status_code} - {response.text}"


@app.route('/')
def home():
    return render_template('index.html')

# Define an endpoint for querying
@app.route('/api/chat', methods=['POST'])
def query():
    try:
        # Get the query from the user input
        user_query = request.json['query']

        # Generate response based on the query
        response = generate_answer(user_query)

        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

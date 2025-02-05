import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template

from api_request_handler import call_llm_api


model = SentenceTransformer('sentence-transformers/all-MiniLM-l6-v2')
with open("docstore.pkl", "rb") as f:
    docstore = pickle.load(f)

# Flask app setup
app = Flask(__name__)

index_i = faiss.read_index("faiss_index.idx")


# function to get relevant documents and generate answers
def generate_answer(user_query):
    query_embedding = model.encode([user_query])[0].astype("float32")
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize query
    D, I = index_i.search(np.array([query_embedding]), k=4)
    relevant_doc = docstore[str(I[0][0])]
    context = relevant_doc['description']
    system_prompt = f""" response user query with provided contex,
    context:{context},
    user query:{user_query}
    """
    response_text = call_llm_api(system_prompt)
    if "</think>" in response_text:
        return response_text.split("</think>")[-1].strip()  # Get text after </think>
    return response_text.strip()  # If </think> not found, return full response


@app.route('/')
def home():
    return render_template('index.html')


# Define an endpoint for querying
@app.route('/api/chat', methods=['POST'])
def query():
    try:
        user_query = request.json['query']
        response = generate_answer(user_query)
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

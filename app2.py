# import numpy as np
# import faiss
# import pickle
# from sentence_transformers import SentenceTransformer
# from flask import Flask, request, jsonify,render_template
#
# from api_request_handler import call_llm_api
#
#
# model = SentenceTransformer('sentence-transformers/all-MiniLM-l6-v2')
# with open("docstore.pkl", "rb") as f:
#     docstore = pickle.load(f)
#
# # Flask app setup
# app = Flask(__name__)
#
# index_i = faiss.read_index("faiss_index.idx")
#
# conversation_history = {}
#
#
# # Define a function to get relevant documents and generate answers
# def generate_answer(user_query, user_id="USER_1"):
#     if user_id not in conversation_history:
#         conversation_history[user_id] = []
#     query_embedding = model.encode([user_query])[0].astype("float32")
#     query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize query
#     D, I = index_i.search(np.array([query_embedding]), k=4)  # Get top 4 relevant documents
#     relevant_doc = docstore[str(I[0][0])]
#     document_context = relevant_doc['description']
#     past_convo = "\n".join(conversation_history[user_id][-1:])
#
#     PROMPT_TEMPLATE = """Answer the question based on the context:
#         User query: {user_query}
#         Context: {document_context}"""
#
#     formatted_prompt = PROMPT_TEMPLATE.format(user_query=user_query, document_context=document_context)
#
#     response_text = call_llm_api(user_query, document_context, formatted_prompt)
#     conversation_history[user_id].append(f"User: {query}")
#     conversation_history[user_id].append(f"""Chatbot: {response_text.split("</think>")[-1].strip()}""")
#     if "</think>" in response_text:
#         return response_text.split("</think>")[-1].strip()  # Get text after </think>
#     return response_text.strip()  # If </think> not found, return full response
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# # Define an endpoint for querying
# @app.route('/api/chat', methods=['POST'])
# def query():
#     try:
#         data = request.get_json()
#         user_query = data.get('user_query')
#         response = generate_answer(user_query)
#         return jsonify({'response': response}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400
#
#
# # Run the Flask app
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001, debug=True)

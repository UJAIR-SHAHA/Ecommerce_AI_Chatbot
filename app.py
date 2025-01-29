from flask import Flask, request, jsonify, render_template
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os

app = Flask(__name__)
index_path = "faiss_index"

# Check if FAISS index exists
if os.path.exists(index_path):
    print("✅ FAISS index file found, proceeding with loading...")
else:
    print("❌ Error: FAISS index file NOT found! Make sure you have saved it correctly.")

# Load embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Use FAISS to load the local index
vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Load LLM (Zephyr model)
model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create the text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

# Create the LLM pipeline using HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Create RAG (Retrieval-Augmented Generation) chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data["query"]
    # Get the response from the RAG chain
    response = rag_chain(user_query)
    return jsonify({"response": response["result"]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

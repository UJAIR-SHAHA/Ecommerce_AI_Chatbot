# Intelligent eCommerce Assistant

This project is a Flask-based eCommerce assistant powered by Hugging Face's AI model. It helps users get relevant product information based on product tags provided by an eCommerce dataset. The assistant responds to user queries about products and provides concise, accurate answers. The system uses product descriptions and features for generating responses.

## Features

- **Product Information Retrieval**: Extracts relevant product information based on user queries.
- **Retrieval-Augmented Generation (RAG)**: Enhances response quality by retrieving relevant data from a vector database before generating answers.
- **Embedding and Search**: Utilizes FAISS for efficient similarity search to find the most relevant product descriptions.
- **AI-Powered Responses**: Uses Hugging Face's models for generating answers based on the query and retrieved context.
- **Custom Frontend**: The frontend is built using HTML, CSS, and JavaScript to provide a user-friendly interface.

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **AI Model**: Hugging Face (DeepSeek-R1-Distill-Qwen-32B or another selected model)
- **Retrieval-Augmented Generation (RAG)**: Uses FAISS to retrieve relevant information before generating a response.
- **Embedding & Search**: Sentence Transformers, FAISS (Facebook AI Similarity Search)
- **Vector Store**: FAISS to store and index embedded product descriptions for fast retrieval.
- **Environment Variables**: `.env` file for storing sensitive data like API keys
- **Requests**: For making HTTP requests to Hugging Face API

## Retrieval-Augmented Generation (RAG) & Vector Store

The assistant follows the **RAG** approach to enhance response accuracy:
1. **Embedding Product Descriptions**:  
   - Each product description is converted into a numerical vector representation using **Sentence Transformers**.
   - These embeddings capture the semantic meaning of the product tags and descriptions.

2. **Storing in FAISS Vector Store**:  
   - The embedded vectors are indexed and stored in FAISS for efficient similarity search.

3. **Retrieving Relevant Data**:  
   - When a user asks a query, the system searches for the most relevant product embeddings using **FAISS similarity search**.
   - The top retrieved results serve as **context** for response generation.

4. **Generating AI Response**:  
   - The retrieved information is passed to the **Hugging Face model**, which generates a concise and relevant response.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. **Install Dependencies**:
   - Install Python dependencies using `pip`:
     ```bash
     pip install -r requirements.txt
     ```

3. **Set Up Environment Variables**:
   - Create a `.env` file in the project root and add your Hugging Face API key:
     ```bash
     HF_TOKEN=<your-hugging-face-api-key>
     ```

4. **Run the Flask Application**:
   - Start the Flask development server:
     ```bash
     python app.py
     ```
   - The app will be available at `http://127.0.0.1:5000/` in your browser.

## Frontend

The frontend is designed using HTML, CSS, and JavaScript to provide a simple interface where users can input queries. The frontend communicates with the Flask backend to retrieve answers based on product descriptions and tags.

### Example Query

- **User Query**: "Tell me about the Titan watch"
- **Response**: "The Titan watch is a men's accessory designed for casual use, particularly suitable for winter. It features a sleek black color, making it a versatile timepiece that can be worn in various casual settings."

## Structure of the Project

```
/project-folder
│___/model
|   |__ product_data.csv
│
├── /static
│   ├── /css
│   ├── /js
│
├── /templates
│   └── index.html
│
├── app.py
├── faiss_index.pkl  # Serialized FAISS vector store
├── embedding_model.py  # Script for embedding and storing product descriptions
├── requirements.txt
├── .env
├── README.md
```

## Usage

1. Visit the homepage (`/`) and input a query in the provided input field.
2. The Flask backend processes the query and retrieves the most relevant product information using FAISS.
3. The AI assistant generates a response based on the retrieved data.
4. The user receives a concise and accurate answer related to the product.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Hugging Face for their AI models and API.
- FAISS for efficient similarity search.
- Sentence Transformers for generating high-quality embeddings.
```
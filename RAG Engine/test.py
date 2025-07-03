import fitz
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load environment variables
load_dotenv()

# Configuration for different API providers
API_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv('API_KEY'),  # Your existing key
        "model": "llama3-8b-8192"  # or "mixtral-8x7b-32768"
    },
    "huggingface": {
        "base_url": "https://api-inference.huggingface.co/v1",
        "api_key": "hf_YOUR_TOKEN_HERE",  # Get from huggingface.co/settings/tokens
        "model": "microsoft/DialoGPT-medium"
    },
    "openai_free": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-YOUR_OPENAI_KEY_HERE",  # Get from platform.openai.com
        "model": "gpt-3.5-turbo"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",  # Local Ollama server
        "api_key": "ollama",  # Ollama doesn't need real key
        "model": "llama2"  # or "mistral", "codellama", etc.
    }
}

# Choose your provider here
CURRENT_PROVIDER = "groq"  # Change to: "huggingface", "openai_free", or "ollama"

# Initialize OpenAI client with selected provider
provider_config = API_PROVIDERS[CURRENT_PROVIDER]
client = OpenAI(
    base_url=provider_config["base_url"],
    api_key=provider_config["api_key"]
)

# Flask app
app = Flask(__name__)
CORS(app)

# Initialize model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

class RAGSystem:
    def __init__(self):
        self.index = None
        self.chunks = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_text_from_pdf(self, path):
        all_text = ""
        try:
            doc = fitz.open(path)
            for page in doc:
                text = page.get_text()
                all_text += text + "\n"
            doc.close()
            return all_text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None
    
    def chunk_text(self, pdf_text, chunk_size=300, overlap=50):
        words = pdf_text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = words[i:i+chunk_size]
            chunks.append(" ".join(chunk))
        return chunks
    
    def build_and_save_embeddings(self, chunks):
        try:
            chunk_embeddings = self.model.encode(chunks)
            embedding_array = np.array(chunk_embeddings).astype("float32")
            dimension = embedding_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embedding_array)
            os.makedirs("data", exist_ok=True)
            faiss.write_index(index, "data/vector.index")
            with open("data/chunks.json", "w") as f:
                json.dump(chunks, f)
            print("✅ FAISS index and chunks saved!")
            return True
        except Exception as e:
            print(f"Error building embeddings: {e}")
            return False
    
    def load_faiss_and_chunks(self, index_path="data/vector.index", chunks_path="data/chunks.json"):
        try:
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                self.index = faiss.read_index(index_path)
                with open(chunks_path, "r") as f:
                    self.chunks = json.load(f)
                print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")
                print(f"✅ Loaded {len(self.chunks)} chunks")
                return True
            else:
                print("Index or chunks file not found")
                return False
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return False
    
    def semantic_search(self, query, top_k=5):
        if self.index is None or self.chunks is None:
            return []
        try:
            query_embedding = self.model.encode([query]).astype("float32")
            distances, indices = self.index.search(query_embedding, top_k)
            relevant_chunks = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    relevant_chunks.append({
                        'chunk': self.chunks[idx],
                        'distance': float(distances[0][i]),
                        'index': int(idx)
                    })
            return relevant_chunks
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def answer_with_llm(self, question, context_chunks):
        try:
            context = "\n\n".join([chunk['chunk'] for chunk in context_chunks])
            prompt = f"""You are an expert tutor in AP Physics C. Use the provided context to answer the question accurately and concisely.

Context: {context}

Question: {question}

Answer:"""

            # Provider-specific model selection
            model_name = provider_config["model"]
            
            # Adjust parameters based on provider
            if CURRENT_PROVIDER == "groq":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.2
                )
            elif CURRENT_PROVIDER == "huggingface":
                # Hugging Face might need different parameters
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.2
                )
            elif CURRENT_PROVIDER == "openai_free":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.2
                )
            elif CURRENT_PROVIDER == "ollama":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.2
                )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating answer with {CURRENT_PROVIDER}: {e}")
            return f"Sorry, I encountered an error while generating the answer using {CURRENT_PROVIDER}."
    
    def initialize_from_pdf(self, pdf_path):
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return False
        chunks = self.chunk_text(pdf_text)
        return self.build_and_save_embeddings(chunks)

# Initialize RAG system
rag_system = RAGSystem()

# Load existing index or create new one
if not rag_system.load_faiss_and_chunks():
    print("No existing index found. Initializing from PDF...")
    pdf_path = "./data/ap-physics-c-mechanics-course-and-exam-description.pdf"
    if rag_system.initialize_from_pdf(pdf_path):
        rag_system.load_faiss_and_chunks()
    else:
        print("Failed to initialize RAG system")

# Flask API Routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "RAG system is running",
        "provider": CURRENT_PROVIDER,
        "model": provider_config["model"]
    })

@app.route('/query', methods=['POST'])
def query_rag():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Question is required"}), 400
        
        question = data['question']
        top_k = data.get('top_k', 5)
        
        relevant_chunks = rag_system.semantic_search(question, top_k)
        if not relevant_chunks:
            return jsonify({"error": "No relevant context found"}), 404
        
        answer = rag_system.answer_with_llm(question, relevant_chunks)
        
        return jsonify({
            "question": question,
            "answer": answer,
            "provider": CURRENT_PROVIDER,
            "model": provider_config["model"],
            "relevant_chunks": [
                {
                    "chunk": chunk['chunk'][:200] + "..." if len(chunk['chunk']) > 200 else chunk['chunk'],
                    "distance": chunk['distance']
                }
                for chunk in relevant_chunks
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/switch_provider', methods=['POST'])
def switch_provider():
    """Switch between different API providers"""
    global CURRENT_PROVIDER, client, provider_config
    
    try:
        data = request.get_json()
        new_provider = data.get('provider')
        
        if new_provider not in API_PROVIDERS:
            return jsonify({"error": f"Invalid provider. Available: {list(API_PROVIDERS.keys())}"}), 400
        
        CURRENT_PROVIDER = new_provider
        provider_config = API_PROVIDERS[CURRENT_PROVIDER]
        
        # Reinitialize client with new provider
        client = OpenAI(
            base_url=provider_config["base_url"],
            api_key=provider_config["api_key"]
        )
        
        return jsonify({
            "message": f"Switched to {CURRENT_PROVIDER}",
            "provider": CURRENT_PROVIDER,
            "model": provider_config["model"]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/providers', methods=['GET'])
def list_providers():
    """List all available API providers"""
    return jsonify({
        "current_provider": CURRENT_PROVIDER,
        "available_providers": {
            name: {"model": config["model"], "base_url": config["base_url"]} 
            for name, config in API_PROVIDERS.items()
        }
    })

# Other existing routes...
@app.route('/search', methods=['POST'])
def search_chunks():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        top_k = data.get('top_k', 10)
        
        relevant_chunks = rag_system.semantic_search(query, top_k)
        
        return jsonify({
            "query": query,
            "chunks": [
                {
                    "chunk": chunk['chunk'],
                    "distance": chunk['distance'],
                    "index": chunk['index']
                }
                for chunk in relevant_chunks
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        if rag_system.index and rag_system.chunks:
            return jsonify({
                "total_vectors": int(rag_system.index.ntotal),
                "total_chunks": len(rag_system.chunks),
                "embedding_dimension": rag_system.index.d,
                "model_name": "all-MiniLM-L6-v2",
                "current_provider": CURRENT_PROVIDER,
                "llm_model": provider_config["model"]
            })
        else:
            return jsonify({"error": "System not initialized"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 Starting RAG system with provider: {CURRENT_PROVIDER}")
    print(f"📊 Using model: {provider_config['model']}")
    app.run(debug=True, host='0.0.0.0', port=os.getenv('PORT'))

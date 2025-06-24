import fitz
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS

# get env variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# make flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# initialize model for embeddings
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
            #error handling
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
        #finding relevant chunks
        if self.index is None or self.chunks is None:
            return []
        
        try:
            # encode prompt
            query_embedding = self.model.encode([query]).astype("float32")
            
            #search using faiss
            distances, indices = self.index.search(query_embedding, top_k)
            
            # find relevant chnks
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
        #get answer
        try:
            context = "\n\n".join([chunk['chunk'] for chunk in context_chunks])
            
            prompt = f"""You are an expert tutor in AP Physics C. Use the provided context to answer the question accurately and concisely.
            
Context: {context}

Question: {question}

Answer:"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.2,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while generating the answer."
    
    def initialize_from_pdf(self, pdf_path):
        #start rag system
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return False
        chunks = self.chunk_text(pdf_text)
        return self.build_and_save_embeddings(chunks)

# initialize RAG system
rag_system = RAGSystem()


#checking for errors in initalization
if not rag_system.load_faiss_and_chunks():
    print("No existing index found. Initializing from PDF...")
    pdf_path = "/Users/neilthakkar/Downloads/ap-physics-c-mechanics-course-and-exam-description-2.pdf"
    if rag_system.initialize_from_pdf(pdf_path):
        rag_system.load_faiss_and_chunks()
    else:
        print("Failed to initialize RAG system")

# Flask API Routes
@app.route('/health', methods=['GET'])
def health_check():
    #checking whether flask api works
    return jsonify({"status": "healthy", "message": "RAG system is running"})

@app.route('/query', methods=['POST'])
def query_rag():
    #this is the function to which the API works
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

# @app.route('/search', methods=['POST'])
# def search_chunks():
#     """Search for relevant chunks without generating answer"""
#     try:
#         data = request.get_json()
        
#         if not data or 'query' not in data:
#             return jsonify({"error": "Query is required"}), 400
        
#         query = data['query']
#         top_k = data.get('top_k', 10)
        
#         relevant_chunks = rag_system.semantic_search(query, top_k)
        
#         return jsonify({
#             "query": query,
#             "chunks": [
#                 {
#                     "chunk": chunk['chunk'],
#                     "distance": chunk['distance'],
#                     "index": chunk['index']
#                 }
#                 for chunk in relevant_chunks
#             ]
#         })
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/reload', methods=['POST'])
# def reload_index():
#     try:
#         if rag_system.load_faiss_and_chunks():
#             return jsonify({"message": "Index reloaded successfully"})
#         else:
#             return jsonify({"error": "Failed to reload index"}), 500
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/stats', methods=['GET'])
# def get_stats():
#     try:
#         if rag_system.index and rag_system.chunks:
#             return jsonify({
#                 "total_vectors": int(rag_system.index.ntotal),
#                 "total_chunks": len(rag_system.chunks),
#                 "embedding_dimension": rag_system.index.d,
#                 "model_name": "all-MiniLM-L6-v2"
#             })
#         else:
#             return jsonify({"error": "System not initialized"}), 500
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
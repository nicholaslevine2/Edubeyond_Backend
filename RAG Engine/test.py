import fitz
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

model = SentenceTransformer('all-MiniLM-L6-v2')

path = "/Users/neilthakkar/Downloads/ap-physics-c-mechanics-course-and-exam-description-2.pdf"
def extract_text_from_pdf(path):
    all_text = " "
    doc = fitz.open(path)
    for page in doc:
        text = page.get_text()
        all_text += text + "\n"
    doc.close()
    return all_text
pdf_text = extract_text_from_pdf(path)
print(pdf_text[:1000])

def chunk_text(pdf_text, chunk_size = 300, overlap = 50):
    words = pdf_text.split()
    chunks = []

    for i in range (0, len(words), chunk_size - overlap):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))

    return chunks

chunks = chunk_text(pdf_text)

chunk_embeddings = model.encode(chunks)

def build_and_save_embeddings(chunk_embeddings, chunks):

    embedding_array = np.array(chunk_embeddings).astype("float32")
    dimension = embedding_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_array)
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/vector.index")

    with open("data/chunks.json", "w") as f:
        json.dump(chunks, f)

    print("✅ FAISS index and chunks saved!")
build_and_save_embeddings(chunk_embeddings, chunks)

def load_faiss_and_chunks(index_path = "data/vector.index", chunks_path = "data/chunks.json"):
    index = faiss.read_index(index_path)
    with open(chunks_path, "r") as f:
        chunks = json.load(f)



    return index, chunks

index, chunks = load_faiss_and_chunks()

def answer_with_llm(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
    You are an expert tutor in AP physics C, and you will use the resources provided to answer questions.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "text": prompt}],
        max_tokens = 300,
        temperature = 0.2,
    )
    return response["choices"][0]["message"]["content"].strip()




print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
print(f"✅ Loaded {len(chunks)} chunks")
print(f"Number of embeddings: {len(chunk_embeddings)}")
print(f"Length of one embedding: {len(chunk_embeddings[0])}")
print("First few numbers of first embedding:", chunk_embeddings[0][:5])
print(f"Total chunks: {len(chunks)}")
print("Sample chunk:\n", chunks[200])

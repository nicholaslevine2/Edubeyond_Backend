import openai
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from test import RAGSystem




load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

def generate_questions_from_chunks(user_prompt, chunks, temperature = 0.7):
    try:
        context = "\n\n".join(chunks)

        prompt =  f"""The user said: "{user_prompt}" 
    Using the following context from textbooks + databases, generate  challenging question question depending on he users request, MCQ or FRQ(Free Response Question):
    Only return the questions, not the answers, unless speciified. Give detailed options for the questions, and also if asked, give the answer with detailed explaination. 
    
Context: {context}"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = [{"role":"user", "content":"prompt"}],
            max_tokens = 350,
            temperature = temperature,
        )

        return response.choices[0].message.content.strip().split("\n")

    except Exception as e:
        print(f"Error generating questions: {e}")
        return ["Sorry, there was an error generating questions."]

@app.route("/questions",methods=["POST"])
def generate_questions_api():
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "prompt is required"}), 400

        user_prompt = data["prompt"]
        top_k = data.get('top_k', 5)

        relevant = RAGSystem.semantic_search(user_prompt, top_k)
        chunks = [c['chunk'] for c in relevant]

        questions = generate_questions_from_chunks(user_prompt, chunks)
        return jsonify({"prompt": user_prompt, "questions": questions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
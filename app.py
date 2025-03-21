from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

# Hugging Face API Details
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

@app.route("/process_text", methods=["POST"])
def process_text():
    data = request.get_json()
    extracted_text = data.get("text", "").strip()

    if not extracted_text:
        return jsonify({"error": "No text provided!"})

    # Send extracted text to Mistral AI
    payload = {"inputs": extracted_text, "parameters": {"max_length": 500}}
    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        ai_response = response.json()[0]["generated_text"]
    else:
        ai_response = f"API Error: {response.text}"

    return jsonify({"extracted_text": extracted_text, "ai_response": ai_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

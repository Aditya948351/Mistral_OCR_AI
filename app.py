from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import requests
import io
import os

app = Flask(__name__)

# Hugging Face API Details (Do NOT expose API key publicly)
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

# Function to process image with OCR and send extracted text to AI
def process_image(image):
    # Perform OCR using Tesseract
    extracted_text = pytesseract.image_to_string(image)

    if not extracted_text.strip():
        return {"error": "No text found!"}

    # Send extracted text to Mistral-7B
    payload = {"inputs": extracted_text.strip(), "parameters": {"max_length": 500}}
    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        ai_response = response.json()[0]["generated_text"]
    else:
        ai_response = f"API Error: {response.text}"

    return {"extracted_text": extracted_text, "ai_response": ai_response}

# Flask Route to Handle Image Upload
@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded!"})

    image_file = request.files["image"]
    image = Image.open(image_file)

    result = process_image(image)

    return jsonify(result)


# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

from flask import Flask, render_template, request, jsonify
import pytesseract
from PIL import Image
import requests
import io
import base64

app = Flask(__name__)

# Hugging Face API Details
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": "Bearer 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"}

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

# Flask Route for Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Flask Route to Handle Image Upload
@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded!"})

    image_file = request.files["image"]
    image = Image.open(image_file)

    result = process_image(image)
    return jsonify(result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

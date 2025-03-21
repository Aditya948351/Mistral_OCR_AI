from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import requests
import io
import os

app = Flask(__name__)

# Hugging Face API Details
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

def process_image(image):
    try:
        # Perform OCR using Tesseract
        extracted_text = pytesseract.image_to_string(image)

        if not extracted_text.strip():
            return {"error": "No text found!"}

        # Debugging: Print extracted text
        print("Extracted Text:", extracted_text)

        # Send extracted text to Mistral-7B
        payload = {"inputs": extracted_text.strip(), "parameters": {"max_length": 500}}
        response = requests.post(API_URL, headers=HEADERS, json=payload)

        # Debugging: Print API response
        print("Mistral API Response Code:", response.status_code)
        print("Mistral API Response Body:", response.text)

        if response.status_code != 200:
            return {"error": f"API Error: {response.text}"}

        ai_response = response.json()[0].get("generated_text", "No response from AI")

        return {"extracted_text": extracted_text, "ai_response": ai_response}

    except Exception as e:
        print("Error processing image:", str(e))
        return {"error": str(e)}

@app.route("/process", methods=["POST"])
def process():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded!"})

        image_file = request.files["image"]
        image = Image.open(image_file)

        result = process_image(image)
        return jsonify(result)

    except Exception as e:
        print("Server Error:", str(e))
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

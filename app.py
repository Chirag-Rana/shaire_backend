from flask import Flask, request, jsonify
#import google.generativeai as genai
from google import genai
from google.genai import types
import cv2
import json
import os
import re
import numpy as np
from PIL import Image

app = Flask(__name__)

# Configure with your API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
client = genai.Client(api_key=GEMINI_API_KEY)

def extract_bill_info_gemini(image_array):
    if image_array is None:
        return {"error": "No image provided"}
    
    try:
        #model = "gemini-2.0-flash"

        prompt = """
        You are an expert at extracting information from receipts and bills.
        Analyze the provided image and extract the following information:
        - Merchant name (restaurant or store name)
        - Date of purchase
        - Total amount
        - All items purchased with their individual amounts

        Return the information in a JSON format with the following structure:
        {
          "merchant_name": "RESTAURANT NAME",
          "date": "YYYY-MM-DD",
          "total_amount": 45.67,
          "items": [
            {
              "description": "Item name",
              "amount": 12.99
            },
            {
              "description": "Another item",
              "amount": 5.99
            }
          ]
        }

        If you cannot find some information, use null or empty values. Output ONLY the JSON.
        Ensure the JSON is valid and can be parsed by a computer.
        """
        
        image_part = {"mime_type": "image/png", "data": cv2.imencode(".png", image_array)[1].tobytes()}
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image_part]
        )
        #response = model.generate_content([prompt, image_part])
        
        json_string = response.text
        
        try:
            data = json.loads(json_string)
            return data
        except json.JSONDecodeError:
            # Cleanup and try again
            json_string = re.sub(r"``````", "", json_string).strip()
            json_match = re.search(r"\{.*\}", json_string, re.DOTALL)
            
            if json_match:
                json_string = json_match.group(0)
                try:
                    data = json.loads(json_string)
                    return data
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON format after cleanup"}
            else:
                return {"error": "No JSON found in response"}
    except Exception as e:
        return {"error": f"Error during Gemini API call: {str(e)}"}

@app.route('/')
def index():
    return jsonify({"message": "Bill extraction API is running. Use /extract_bill to upload an image."})

@app.route('/extract_bill', methods=['POST'])
def extract_bill():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        
        # Read the image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Process the image
        result = extract_bill_info_gemini(img)
        
        return jsonify(result), 200
    
    except Exception as e:
        print(f"Error: {str(e)}")  # Log error for debugging
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check external service availability (e.g., Gemini API key validation)
        #genai.GenerativeModel('gemini-2.0-flash')  # Test model initialization
        client.models.get_model(model="gemini-2.0-flash")
        return jsonify({"status": "healthy", "message": "Server is running"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    return jsonify({
        "merchant_name": "TEST RESTAURANT",
        "date": "2023-03-08",
        "total_amount": 45.67,
        "items": [
            {
                "description": "Burger",
                "amount": 12.99
            },
            {
                "description": "Fries",
                "amount": 5.99
            },
            {
                "description": "Soda",
                "amount": 2.49
            }
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8000)))

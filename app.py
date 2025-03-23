from flask import Flask, request, jsonify
import requests
import pandas as pd
import spacy
from dotenv import load_dotenv
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

os.system("python -m spacy download en_core_web_sm")  # Ensure model is available
nlp = spacy.load("en_core_web_sm")

# Hugging Face API details for text simplification
API_URL1 = "https://api-inference.huggingface.co/models/JexCaber/TransLingo"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}
MODEL_NAME1 = "JexCaber/TransLingo-Terms2"
tokenizer1 = AutoTokenizer.from_pretrained(MODEL_NAME1)
model1 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME1)

def split_by_topics(text):
    doc = nlp(text)
    topic_chunks = []
    current_chunk = []
    current_entities = set()

    for sent in doc.sents:
        entities = {ent.text.lower() for ent in sent.ents}
        if current_entities and len(current_entities.intersection(entities)) < 1:
            topic_chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_entities = entities
        else:
            current_entities.update(entities)
        current_chunk.append(sent.text)

    if current_chunk:
        topic_chunks.append(" ".join(current_chunk))
    
    if not topic_chunks:
        topic_chunks.append(text)  # Ensure there is at least one chunk
    
    return topic_chunks

def simplify_text(paragraph):
    topic_chunks = split_by_topics(paragraph)
    simplified_output = {}

    for i, chunk in enumerate(topic_chunks):
        generation_params = {
            "inputs": chunk,
            "parameters": {
                "max_length": 150,
                "do_sample": True,  # Fix: Ensure sampling is enabled
                "temperature": 0.7
            }
        }
        
        try:
            response = requests.post(API_URL1, headers=HEADERS, json=generation_params)
            print("HF API Response Status:", response.status_code)  # Debugging
            print("HF API Response Text:", response.text)  # Debugging
            response.raise_for_status()
            api_response = response.json()

            if isinstance(api_response, list) and "generated_text" in api_response[0]:
                simplified_chunk = api_response[0]["generated_text"]
            else:
                return {"error": "Unexpected API response", "details": api_response}
        
        except requests.exceptions.RequestException as e:
            return {"error": "Request to Hugging Face API failed", "details": str(e)}
        
        simplified_output[f"Topic {i+1}"] = simplified_chunk
    
    return simplified_output

@app.route("/simplify-text", methods=['POST'])
def api_simplify_text():
    try:
        data = request.json
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        simplified_text = simplify_text(text)
        return jsonify(simplified_text)

    except Exception as e:
        print("Error in /simplify-text endpoint:", str(e))  # Debugging
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Welcome to the TransLingo API!"

port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port or default to 10000
app.run(host="0.0.0.0", port=port)
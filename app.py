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

def nested_sliding_window_split(text, min_size=3, max_size=12, step_size=5):
    """Splits text into overlapping word chunks of varying sizes."""
    words = text.split()
    chunks = []
    
    for size in range(min_size, max_size + 1, step_size):  
        for i in range(0, len(words) - size + 1, step_size):
            chunk = " ".join(words[i:i+size])
            chunks.append(chunk)
    
    return chunks

def clean_term_output(output_text):
    """Cleans extracted term and removes redundant 'Term: Term:' patterns."""
    output_text = output_text.replace("Term: Term:", "Term:").replace("Term:", "", 1).strip()
    return output_text

def extract_terms_from_paragraph(paragraph):
    chunks = nested_sliding_window_split(paragraph)
    extracted_terms = {}

    for chunk in chunks:
        input_ids = tokenizer1.encode(chunk, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            output_ids = model1.generate(
                input_ids, 
                max_length=512, 
                num_beams=3,  
                temperature=0.7,  
                repetition_penalty=1.2  
            )

        output_text = tokenizer1.decode(output_ids[0], skip_special_tokens=True)
        term, definition = output_text.split("| Definition: ", 1) if "| Definition: " in output_text else (output_text, "")

        term = clean_term_output(term).lower()  # Normalize term
        
        # Avoid duplicate terms
        if term and term not in extracted_terms:
            extracted_terms[term] = definition.strip()

    return extracted_terms  

def split_by_topics(text):
    doc = nlp(text)
    topic_chunks = []
    current_chunk = []
    current_entities = set()

    for sent in doc.sents:
        entities = {ent.text.lower() for ent in sent.ents}  # Extract Named Entities

        # If new entities appear, start a new topic
        if current_entities and len(current_entities.intersection(entities)) < 1:
            topic_chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_entities = entities
        else:
            current_entities.update(entities)

        current_chunk.append(sent.text)

    if current_chunk:
        topic_chunks.append(" ".join(current_chunk))

    return topic_chunks

# Function to simplify text using the Hugging Face API
def simplify_text(paragraph):
    topic_chunks = split_by_topics(paragraph)  # Step 1: Split topics
    simplified_output = {}

    for i, chunk in enumerate(topic_chunks):
        generation_params = {
            "inputs": chunk,
            "parameters": {
                "max_length": 150
            }
        }

        # Request to Hugging Face API
        response = requests.post(API_URL1, headers=HEADERS, json=generation_params)
        response.raise_for_status()  # Raise error if request fails

        # Extract response
        simplified_chunk = response.json()[0]['generated_text']
        simplified_output[f"Topic {i+1}"] = simplified_chunk

    return simplified_output

@app.route("/simplify-text", methods=['POST'])
def api_simplify_text():
    try:
        data = request.json
        text = data.get('text')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Process simplification
        simplified_text = simplify_text(text)

        return jsonify(simplified_text)

    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route("/term-detection", methods=['POST'])
def term_detection():
    try:
        data = request.json
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        extracted_terms = extract_terms_from_paragraph(text)
        
        return jsonify({"extracted_terms": extracted_terms})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Welcome to the TransLingo API!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
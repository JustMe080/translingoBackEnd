from flask import Flask, request, jsonify
import requests
import pandas as pd
import spacy
from dotenv import load_dotenv
from flask_cors import CORS
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
API_URL2 = "https://api-inference.huggingface.co/models/JexCaber/TransLingo-Terms2"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_TOKEN')}"}



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
    """Cleans extracted term and removes redundant patterns."""
    output_text = output_text.replace("Term: Term:", "Term:").replace("Term:", "", 1).strip()
    return output_text

def extract_terms_from_paragraph(paragraph):
    """Uses Hugging Face API to detect terms in text."""
    chunks = nested_sliding_window_split(paragraph)
    extracted_terms = {}

    for chunk in chunks:
        generation_params = {
            "inputs": chunk,
            "parameters": {"max_length": 512}
        }

        try:
            response = requests.post(API_URL2, headers=HEADERS, json=generation_params)
            response.raise_for_status()
            output_text = response.json()[0]['generated_text']

            term, definition = output_text.split("| Definition: ", 1) if "| Definition: " in output_text else (output_text, "")
            term = clean_term_output(term).lower()

            if term and term not in extracted_terms:
                extracted_terms[term] = definition.strip()

        except requests.exceptions.RequestException as e:
            return {"error": "Hugging Face API request failed", "details": str(e)}

    return extracted_terms

def split_by_topics(text):
    """Splits text into topic-based chunks using Named Entity Recognition (NER)."""
    doc = nlp(text)
    topic_chunks = []
    current_chunk = []
    current_entities = set()

    for sent in doc.sents:
        entities = {ent.text.lower() for ent in sent.ents}

        # Start a new topic if entities change significantly
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

def simplify_text(paragraph):
    """Uses Hugging Face API to simplify text based on topic chunks."""
    topic_chunks = split_by_topics(paragraph)
    simplified_output = {}

    for i, chunk in enumerate(topic_chunks):
        generation_params = {
            "inputs": chunk,
            "parameters": {"max_length": 150}
        }

        try:
            response = requests.post(API_URL1, headers=HEADERS, json=generation_params)
            response.raise_for_status()
            simplified_chunk = response.json()[0]['generated_text']
            simplified_output[f"Topic {i+1}"] = simplified_chunk

        except requests.exceptions.RequestException as e:
            return {"error": "Hugging Face API request failed", "details": str(e)}

    return simplified_output

@app.route("/simplify-text", methods=['POST'])
def api_simplify_text():
    """API endpoint for text simplification."""
    try:
        data = request.json
        text = data.get('text')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        simplified_text = simplify_text(text)
        return jsonify(simplified_text)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/term-detection", methods=['POST'])
def term_detection():
    """API endpoint for term detection."""
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
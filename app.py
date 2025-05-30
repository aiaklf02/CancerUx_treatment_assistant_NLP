import re
import torch
import spacy
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import yake

app = Flask(__name__)

# Charger le modèle BioBERT fine-tuné
model_path = "./biobert-squad-cancer"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)

# Charger un modèle spaCy pour NER médical léger
nlp = spacy.load("en_core_web_sm")  # Remplacez par un modèle clinique si disponible


# YAKE pour mots-clés
kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=5)

# Charger contextes

def load_contexts(file_path="squad_cancer_data.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    contexts = []
    for item in data.get("data", []):
        for para in item.get("paragraphs", []):
            contexts.append(para.get("context", ""))
    return contexts

contexts = load_contexts()

# Nettoyage & tokenisation

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # espaces multiples
    text = re.sub(r"\d+", "", text)  # chiffres
    return text.strip()

# Extraire entités médicales

def extract_medical_entities(text):
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents if ent.label_ in ("DISEASE", "CONDITION", "ORG", "GPE", "PERSON")]

# Trouver contexte pertinent

def find_best_context(question, contexts):
    cleaned_q = clean_and_tokenize(question)
    medical_ents = extract_medical_entities(question)
    filtered = [ctx for ctx in contexts if any(ent in ctx.lower() for ent in medical_ents)] or contexts

    vectorizer = TfidfVectorizer().fit([cleaned_q] + filtered)
    vectors = vectorizer.transform([cleaned_q] + filtered)
    sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    best_idx = sims.argmax()
    return filtered[best_idx], medical_ents

# YAKE mots-clés

def extract_keywords(text):
    return [kw for kw, _ in kw_extractor.extract_keywords(text)]

# Routes Flask

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"response": "Please enter a question."})

    # Nettoyage + sélection contexte
    best_ctx, ents = find_best_context(question, contexts)

    # QA
    inputs = tokenizer(question, best_ctx, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    ans_ids = inputs["input_ids"][0][start_idx:end_idx]
    answer = tokenizer.decode(ans_ids, skip_special_tokens=True).strip()

    # mots-clés
    keywords = extract_keywords(answer)

    return jsonify({
        "response": answer,
        "keywords": keywords,
        "entities": ents
    })

if __name__ == '__main__':
    app.run(debug=True)

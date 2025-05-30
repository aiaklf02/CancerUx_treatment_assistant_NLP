from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
import re
from typing import List, Dict, Any

# === 1. Basic Text Cleaning ===
def basic_clean(text: str) -> str:
    """
    Remove extra whitespace, newlines, and normalize text.
    """
    text = re.sub(r"\n+", " ", text)      # collapse newlines
    text = re.sub(r"\s+", " ", text)      # collapse whitespace
    text = re.sub(r"\d+", " ", text)      

    return text.strip()

# === 2. Load & Structure JSON Data ===

def load_and_clean(json_path: str) -> List[Dict[str, Any]]:
    """
    Read the NCCN-derived JSON, clean each entry,
    and return a flat list of dicts with keys:
    - cancer_type
    - section
    - text
    """
    raw = json.load(open(json_path, encoding="utf-8"))

    docs: List[Dict[str, Any]] = []
    for cancer_type, info in raw.items():
        treatments = info.get("treatments", {})
        for section, entries in treatments.items():
            for entry in entries:
                cleaned = basic_clean(entry)
                if cleaned:
                    docs.append({
                        "cancer_type": cancer_type,
                        "section": section,
                        "text": cleaned
                    })
    return docs

# === 3. Initialize Question Generation Model ===
tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-prepend")
model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-small-qg-prepend")

def generate_question(context: str, answer: str) -> str:
    prompt = f"generate question: {answer} context: {context}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === 4. Process and Generate SQuAD-like Dataset ===
docs = load_and_clean("cancer_data.json")
squad_dataset = {"data": []} 
for doc in docs:
    paragraph = doc["text"]
    answer = paragraph.split(".")[0]
    question = generate_question(paragraph, answer)

    squad_dataset["data"].append({
        "title": f"{doc['cancer_type']} – {doc['section']}",
        "paragraphs": [{
            "context": paragraph,
            "qas": [{
                "id": f"{doc['cancer_type']}-{hash(paragraph)}",
                "question": question,
                "answers": [{
                    "text": answer,
                    "answer_start": paragraph.find(answer)
                }],
                "is_impossible": False
            }]
        }]
    })

# === 5. Save to File ===
with open("squad_cancer_data.json", "w", encoding="utf-8") as f:
    json.dump(squad_dataset, f, ensure_ascii=False, indent=2)


from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, default_data_collator
)
import torch
import evaluate
import numpy as np
import json

# Charger le dataset initial et le découper en train/validation
raw_dataset = load_dataset("json", data_files="squad_cancer_data.json", field="data")["train"]
split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)

# Fonction pour reconstruire une liste d’exemples à partir d’un dictionnaire
#partir de dict avec key et liste de valeur en [
  #{"question": "Question 1", "context": "Context 1", "answers": réponse1},
 # {"question": "Question 2", "context": "Context 2", "answers": réponse2},
 # {"question": "Question 3", "context": "Context 3", "answers": réponse3}

def reconstruct_examples(dict_data):
    columns = dict_data.keys()
    n = len(next(iter(dict_data.values())))
    examples = []
    for i in range(n):
        example = {col: dict_data[col][i] for col in columns}
        examples.append(example)
    return examples

# Sauvegarde en fichiers JSON (train et validation)
train_examples = reconstruct_examples(split_dataset["train"].to_dict())
test_examples = reconstruct_examples(split_dataset["test"].to_dict())

with open("squad_cancer_train.json", "w", encoding="utf-8") as f:
    json.dump({"data": train_examples}, f, ensure_ascii=False, indent=2)

with open("squad_cancer_validation.json", "w", encoding="utf-8") as f:
    json.dump({"data": test_examples}, f, ensure_ascii=False, indent=2)

# Initialisation du tokenizer et du modèle BioBERT
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Rechargement du dataset (train + validation)
data_files = {
    "train": "squad_cancer_train.json",
    "validation": "squad_cancer_validation.json"
}
raw_datasets = load_dataset("json", data_files=data_files, field="data")

# Fonction de prétraitement
def preprocess_function(examples):
    contexts = []
    questions = []
    answers = []

    for paragraphs in examples["paragraphs"]:
        for ex in paragraphs:
            context = ex["context"]
            for qa in ex["qas"]:
                contexts.append(context)
                questions.append(qa["question"])
                answers.append(qa["answers"][0])  # première réponse

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])
        sequence_ids = tokenized.sequence_ids(i)

        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if not (offsets[context_start][0] <= start_char and offsets[context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_start_index = context_start
            while token_start_index <= context_end and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            token_end_index = context_end
            while token_end_index >= context_start and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            start_positions.append(token_start_index - 1)
            end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized

# Appliquer le prétraitement
tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# Configuration de l'entraînement
training_args = TrainingArguments(
    output_dir="./biobert-qa-checkpoint",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    fp16=torch.cuda.is_available()
)

# Chargement du metric SQuAD
metric = evaluate.load("squad")

# Fonction de calcul des métriques (simplifiée)
def compute_metrics(p):
    start_preds = np.argmax(p.predictions[0], axis=-1)
    end_preds = np.argmax(p.predictions[1], axis=-1)

    pred_texts = []
    true_texts = []

    for i in range(len(start_preds)):
        offsets = p.inputs["offset_mapping"][i]
        input_ids = p.inputs["input_ids"][i]

        start_pred = start_preds[i]
        end_pred = end_preds[i]

        if start_pred >= len(offsets) or end_pred >= len(offsets):
            pred_texts.append("")
        else:
            start_char = offsets[start_pred][0]
            end_char = offsets[end_pred][1]
            pred_text = tokenizer.decode(input_ids[start_pred:end_pred+1])
            pred_texts.append(pred_text)

        true_texts.append("")  # facultatif : à remplir si tu veux comparer à la vraie réponse

    return metric.compute(predictions=pred_texts, references=true_texts)

# Initialisation du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

# Entraînement
trainer.train()

# Sauvegarde du modèle
trainer.save_model("./biobert-squad-cancer")
tokenizer.save_pretrained("./biobert-squad-cancer")

# Évaluation finale
eval_results = trainer.evaluate()
print("\n📊 Evaluation metrics on validation set:")
for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")
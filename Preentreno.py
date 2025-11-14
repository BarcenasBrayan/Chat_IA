from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import pandas as pd
import torch

# ---------- CARGA TU DATASET ----------
df = pd.read_csv("C:/Users/Brayan/Desktop/Python/Certificacion/Proyecto/intenciones.csv")  # Asegúrate de tener las columnas: texto, etiqueta
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["etiqueta"])

dataset = Dataset.from_pandas(df[["texto", "label"]])
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

def tokenize(batch):
    return tokenizer(batch["texto"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)

# ---------- DIVIDIR EN TRAIN Y TEST ----------
split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

# ---------- CARGAR MODELO ----------
model = BertForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", num_labels=len(label_encoder.classes_))

# ---------- CONFIGURAR ENTRENAMIENTO ----------
training_args = TrainingArguments(
    output_dir="./beto_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ---------- ENTRENAR ----------
trainer.train()


# 1. Obtener predicciones sobre el conjunto de evaluación
predictions = trainer.predict(eval_dataset)

# 2. Convertir logits a etiquetas predichas
y_pred = predictions.predictions.argmax(-1)
y_true = predictions.label_ids

# 3. Nombres de las clases (ajusta según tu mapeo)
label_names = ["info_clave_texto", "recomendacion_peliculas", "sobre_bot", "saludo"]

# 4. Mostrar el reporte de clasificación
print("\n📊 Reporte de clasificación detallado:\n")
print(classification_report(y_true, y_pred, target_names=label_names))


# ---------- GUARDAR ----------
model.save_pretrained("./beto_finetuned")
tokenizer.save_pretrained("./beto_finetuned")
with open("reporte_f1_beto.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(y_true, y_pred, target_names=label_names))

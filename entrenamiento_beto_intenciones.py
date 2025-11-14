from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

# Paso 1: Datos de ejemplo
intenciones = ["ConsultaHora", "Recomendacion", "ConocerBot", "Saludo", "Despedida"]
datos = {
    "texto": [
        "¿Qué hora es?",
        "Recomiéndame una película",
        "¿Quién te creó?",
        "Hola",
        "Gracias",
        "¿Qué hora tienes?",
        "Dime una película buena",
        "¿Cómo fuiste desarrollado?",
        "Buenas tardes",
        "Nos vemos"
    ],
    "label": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
}

# Paso 2: División en entrenamiento y validación
train_texts, val_texts, train_labels, val_labels = train_test_split(
    datos["texto"], datos["label"], test_size=0.2, random_state=42
)

# Paso 3: Crear datasets
train_dataset = Dataset.from_dict({"texto": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"texto": val_texts, "label": val_labels})

# Paso 4: Tokenización
modelo_base = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = BertTokenizer.from_pretrained(modelo_base)

def tokenizar(ejemplo):
    return tokenizer(ejemplo["texto"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenizar)
val_dataset = val_dataset.map(tokenizar)

# Paso 5: Preparar modelo
model = BertForSequenceClassification.from_pretrained(modelo_base, num_labels=len(intenciones))

# Paso 6: Métricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

# Paso 7: Configuración de entrenamiento
training_args = TrainingArguments(
    output_dir="./beto-clasificador-intenciones",
    eval_strategy="epoch",  # <- ya puedes usar esto correctamente
    per_device_train_batch_size=4,
    num_train_epochs=10,
    logging_dir="./logs",
    save_total_limit=1,
    save_strategy="epoch"
)

# Paso 8: Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Paso 9: Entrenamiento
trainer.train()

# Paso 10: Guardado
trainer.save_model("beto-clasificador-intenciones")
tokenizer.save_pretrained("beto-clasificador-intenciones")


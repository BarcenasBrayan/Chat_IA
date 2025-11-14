from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets

# 1. Cargar el tokenizer y modelo base en español
tokenizer = GPT2Tokenizer.from_pretrained("./modelo-local")
model = GPT2LMHeadModel.from_pretrained("./modelo-local")
tokenizer.pad_token = tokenizer.eos_token  # Para evitar errores con padding

tokenizer.add_tokens(["<FIN_SINOPSIS>"]) 
model.resize_token_embeddings(len(tokenizer))
 
# 2. Preparar dataset
def cargar_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

dataset = cargar_dataset("C:/Users/Brayan/Desktop/Python/Certificacion/dataset_sinopsis_estilizadas.txt", tokenizer)
#dataset = cargar_dataset("C:/Users/Brayan/Desktop/Python/Certificacion/saludos_ubuntu.txt", tokenizer)
#combined_dataset = concatenate_datasets([dataset0, dataset1])

# 3. Creador de batches dinámicos
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 no usa Masked LM
)

# 4. Parámetros de entrenamiento
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    weight_decay=0.01
)

# 5. Entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# 6. Iniciar entrenamiento
trainer.train()

# 7. Guardar modelo
trainer.save_model("./gpt2-finetuned")
tokenizer.save_pretrained("./gpt2-finetuned")

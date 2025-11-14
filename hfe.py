from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Descarga los archivos a una carpeta local
tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")
model = GPT2LMHeadModel.from_pretrained("datificate/gpt2-small-spanish")

# Guarda localmente
tokenizer.save_pretrained("./modelo-local")
model.save_pretrained("./modelo-local")

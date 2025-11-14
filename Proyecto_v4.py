import customtkinter as ctk
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import torch
import re

chat_history = ""  # Historial acumulado

def procesar_respuesta(respuesta, prompt, max_chars=800):
    texto = respuesta[len(prompt):].strip()
    texto = re.sub(r'(Usuario:|<\|endoftext\|>)', '', texto, flags=re.IGNORECASE).strip()

    # Cortar si empieza a escribir el siguiente turno del usuario
    if "Usuario:" in texto:
        texto = texto.split("Usuario:")[0].strip()
    if "Enrique:" in texto[1:]:  # Si genera "Enrique:" nuevamente, también cortamos ahí
        texto = texto.split("Enrique:")[0].strip()

    if len(texto) > max_chars:
        texto = texto[:max_chars].rsplit(" ", 1)[0] + "..."

    return texto


# Cargar modelos de Hugging Face
# Asegúrate de que los directorios './beto_finetuned' y './gpt2-finetuned'
# contengan los modelos y tokenizadores pre-entrenados y/o fine-tuned.
try:
    beto_model = BertForSequenceClassification.from_pretrained("./beto_finetuned")
    beto_tokenizer = BertTokenizer.from_pretrained("./beto_finetuned")
    beto_model.eval() # Pone el modelo en modo de evaluación (desactiva dropout, etc.)

    gpt2_model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")
    gpt2_model.eval() # Pone el modelo en modo de evaluación
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token # Configura el token de padding para GPT-2
except Exception as e:
    print(f"Error al cargar los modelos: {e}")
    print("Asegúrate de que los modelos 'beto_finetuned' y 'gpt2-finetuned' estén en el directorio correcto.")
    # Puedes añadir un sys.exit() aquí si los modelos son esenciales para la aplicación
    # import sys
    # sys.exit(1)


# Mapeo de IDs de predicción a etiquetas de intención
id2label = {
    0: "info_clave_texto",
    1: "recomendacion_peliculas",
    2: "sobre_bot",
    3: "saludo"
}

# Plantillas de prompt para cada intención
plantillas = {
    "info_clave_texto": "Usuario: {entrada}\nResumen:",
    "recomendacion_peliculas": "Usuario: {entrada}\nEnrique:",
    "sobre_bot": "Usuario: {entrada}\nEnrique:",
    "saludo": "Usuario: {entrada}\nEnrique:"
}

def responder():
    global chat_history

    entrada = input_entry.get().strip()
    if not entrada:
        return

    # Clasificación de intención con BETO
    tokens = beto_tokenizer(entrada, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        salida = beto_model(**tokens)
        pred = torch.argmax(salida.logits, dim=1).item()
        intencion = id2label[pred]

    # Añadir turno del usuario al historial
    chat_history += f"Usuario: {entrada}\nEnrique:"

    # Recortar historial si es muy largo
    max_tokens = 700  # Por seguridad, mantenerlo debajo de 1024 para GPT-2 small
    tokens_hist = gpt2_tokenizer.encode(chat_history)
    if len(tokens_hist) > max_tokens:
        tokens_hist = tokens_hist[-max_tokens:]
        chat_history = gpt2_tokenizer.decode(tokens_hist)

    # Crear entrada para GPT-2
    gpt_input = gpt2_tokenizer(chat_history, return_tensors="pt")
    with torch.no_grad():
        output = gpt2_model.generate(
            **gpt_input,
            max_length=len(gpt_input["input_ids"][0]) + 500,
            do_sample=True,
            top_p=0.90,
            temperature=0.5,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )

    respuesta = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    respuesta_final = procesar_respuesta(respuesta, chat_history)

    # Añadir respuesta del bot al historial
    chat_history += f" {respuesta_final}\n"

    # Mostrar en la interfaz
    output_box.configure(state="normal")
    output_box.insert("end", f"🧑 Tú: {entrada}\n")
    output_box.insert("end", f"🤖 Enrique: {respuesta_final}\n\n")
    output_box.configure(state="disabled")
    input_entry.delete(0, "end")


# Configuración de la ventana principal de CustomTkinter
ctk.set_appearance_mode("light") # Establece el modo de apariencia a "light"
ctk.set_default_color_theme("blue") # Establece el tema de color por defecto a "blue" (un tema predefinido válido)

app = ctk.CTk() # Crea la ventana principal de la aplicación
app.geometry("700x600") # Establece el tamaño inicial de la ventana
app.title("Enrique - IA Asistente") # Establece el título de la ventana
app.configure(fg_color="orange") # Fondo de la ventana principal naranja

# Crear y configurar la caja de texto para la salida (conversación)
output_box = ctk.CTkTextbox(app, width=650, height=400, font=("Arial", 12), fg_color="salmon") # Fondo salmón
output_box.grid(row=0, column=0, padx=20, pady=20, sticky="nsew") # Posiciona y permite que se expanda
output_box.configure(state="disabled") # La caja de texto comienza deshabilitada

# Crear y configurar el campo de entrada para el usuario
input_entry = ctk.CTkEntry(app, width=500, placeholder_text="Escribe tu mensaje aquí...", text_color="black") # Texto del usuario negro
input_entry.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w") # Posiciona a la izquierda

# Crear y configurar el botón de enviar
send_button = ctk.CTkButton(app, text="Enviar", command=responder, fg_color="black") # Botón negro
send_button.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="e") # Posiciona a la derecha

# Configurar el grid para que la caja de texto se expanda verticalmente
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)

# Iniciar el bucle principal de la aplicación
app.mainloop()
import customtkinter as ctk
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import pandas as pd
import random
import numpy as np
import os # Para manejo de archivos/rutas

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN RAG ---

# Cargar la base de datos de películas
try:
    df_peliculas = pd.read_csv("C:\\Users\\Brayan\\Desktop\\Python\\DatasetPruebas\\peliculas_populares_tmdb.csv")
    
    # 1. Manejar NaNs (nulos) y limpiar datos
    df_peliculas = df_peliculas.fillna({'Año': np.nan, 'Género': '', 'Reseña': 'Sinopsis no disponible.', 'Recomendaciones': ''})
    
    # Asegurar que 'Año' se maneje como entero, reemplazando NaNs con '????' para el display
    df_peliculas['Año_Display'] = df_peliculas['Año'].apply(
        lambda x: str(int(x)) if pd.notna(x) and x > 1900 else '????'
    )
    
    # Creamos un campo de "Puntuación/Popularidad" simulado si no existe
    if 'Puntuacion_Calidad' not in df_peliculas.columns:
         df_peliculas['Puntuacion_Calidad'] = np.random.rand(len(df_peliculas)) * 5 + 5
    
    # Limpiamos los géneros para la búsqueda (se convierten a minúsculas)
    df_peliculas['Género_Lista'] = df_peliculas['Género'].apply(
        lambda x: [g.strip().lower() for g in str(x).split(',')]
    )
    print("Base de datos de películas cargada y limpia correctamente.")
except FileNotFoundError:
    print("ADVERTENCIA: Archivo 'peliculas_populares_tmdb.csv' no encontrado.")
    df_peliculas = pd.DataFrame()
except Exception as e:
    print(f"Error al cargar la base de datos de películas: {e}")
    df_peliculas = pd.DataFrame()


chat_history = ""  # Historial acumulado

def procesar_respuesta(respuesta, prompt, max_chars=800):
    """
    Función para limpiar la respuesta generada por GPT-2 de artefactos 
    (solo para intenciones que NO son recomendación de películas).
    """
    # ... (El código de limpieza se mantiene igual) ...
    texto = respuesta[len(prompt):].strip()
    # Limpiar el marcador de contexto inyectado si se filtró por error
    texto = re.sub(r'Datos_Película:.*', '', texto, flags=re.IGNORECASE).strip() 
    texto = re.sub(r'(Usuario:|<\|endoftext\|>)', '', texto, flags=re.IGNORECASE).strip()

    if "Usuario:" in texto:
        texto = texto.split("Usuario:")[0].strip()
    if "Enrique:" in texto[1:]:
        texto = texto.split("Enrique:")[0].strip()

    if len(texto) > max_chars:
        texto = texto[:max_chars].rsplit(" ", 1)[0] + "..."

    return texto


# Cargar modelos de Hugging Face
try:
    beto_model = BertForSequenceClassification.from_pretrained("./beto_finetuned")
    beto_tokenizer = BertTokenizer.from_pretrained("./beto_finetuned")
    beto_model.eval()

    gpt2_model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")
    gpt2_model.eval()
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
except Exception as e:
    print(f"Error al cargar los modelos: {e}")
    print("Asegúrate de que los modelos 'beto_finetuned' y 'gpt2-finetuned' estén en el directorio correcto.")


# Mapeo de IDs de predicción a etiquetas de intención
id2label = {
    0: "info_clave_texto",
    1: "recomendacion_peliculas",
    2: "sobre_bot",
    3: "saludo"
}

# --- 2. LÓGICA DE RECUPERACIÓN (RAG) SUTIL ---

def obtener_prompt_enriquecido(entrada, chat_history):
    """
    Recupera la información del CSV, crea un prompt enriquecido para GPT-2 
    y retorna los datos factuales de la película.
    """
    global df_peliculas
    
    if df_peliculas.empty:
        return None, None 

    entrada_lower = entrada.lower()
    genero_buscado = None
    generos_disponibles = set(g for sublist in df_peliculas['Género_Lista'] for g in sublist)
    
    # Buscar el género en la entrada
    for gen in generos_disponibles:
        if gen in entrada_lower:
            genero_buscado = gen
            break

    candidatas = df_peliculas

    if genero_buscado:
        candidatas = df_peliculas[df_peliculas['Género_Lista'].apply(lambda x: genero_buscado in x)]
    
    if candidatas.empty:
        # Fallback a las películas más populares si no hay coincidencias de género o no se encontró el género.
        top_n = max(5, int(len(df_peliculas) * 0.3))
        candidatas = df_peliculas.sort_values(by='Puntuacion_Calidad', ascending=False).head(top_n)

    # 1. Seleccionamos la película
    if candidatas.empty:
        return None, None # No hay películas disponibles
        
    pelicula_seleccionada = candidatas.sample(1).iloc[0]
    
    # 2. Recuperación y limpieza de datos Factuales
    titulo = pelicula_seleccionada.get('Película', 'Título Desconocido').replace('"', '').strip()
    anio = pelicula_seleccionada.get('Año_Display', '????').strip()
    genero = pelicula_seleccionada.get('Género', 'Sin Género').strip()
    
    reseña_raw = pelicula_seleccionada.get('Reseña', 'Sinopsis no disponible.').strip()
    # Limpieza estricta de la sinopsis
    reseña_limpia = re.sub(r'[\r\n]+', ' ', reseña_raw).strip()
    reseña_limpia = re.sub(r' +', ' ', reseña_limpia)

    # 3. CONSTRUCCIÓN DEL PROMPT ENRIQUECIDO
    # Inyectamos los datos clave de manera estructurada en el prompt de GPT-2
    contexto_factual = (
        f"Datos_Película: Título=\"{titulo}\" Año={anio} Géneros=\"{genero}\" Sinopsis=\"{reseña_limpia}\""
    )
    
    # Concatenamos el historial, la entrada del usuario y el contexto factual
    # La clave es terminar con "Enrique: " para que GPT-2 genere la introducción.
    prompt_gpt2 = (
        f"{chat_history}"
        f"Usuario: {entrada}\n"
        f"{contexto_factual}\n"
        f"Enrique: "
    )
    
    # Información Factual Clara para la inyección final
    info_recuperada = {
        "titulo": titulo,
        "anio": anio,
        "genero": genero,
        "sinopsis": reseña_limpia
    }
    
    return prompt_gpt2, info_recuperada


# --- 3. FUNCIÓN DE RESPUESTA CON LÓGICA RAG INTEGRADA ---

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
    
    respuesta_final = ""
    
    # --------------------------- LÓGICA RAG SUTIL ---------------------------
    if intencion == "recomendacion_peliculas":
        
        # Recortar historial si es muy largo (mismo código de antes)
        max_tokens = 700 
        tokens_hist = gpt2_tokenizer.encode(chat_history)
        if len(tokens_hist) > max_tokens:
            tokens_hist = tokens_hist[-max_tokens:]
            chat_history = gpt2_tokenizer.decode(tokens_hist)
            
        # 1. Recuperar Prompt y Datos Factuales
        prompt_gpt2, info_recuperada = obtener_prompt_enriquecido(entrada, chat_history)

        if prompt_gpt2 is None:
             respuesta_final = "Lo siento, mi base de datos de películas no está disponible o no hay coincidencias."
        else:
            # 2. GENERACIÓN con el PROMPT ENRIQUECIDO
            gpt_input = gpt2_tokenizer(prompt_gpt2, return_tensors="pt")
            
            with torch.no_grad():
                output = gpt2_model.generate(
                    **gpt_input,
                    temperature=0.6, # Reducir temperatura para más coherencia
                    top_p=0.9,
                    max_length=len(gpt_input["input_ids"][0]) + 40, # Generar SOLO la introducción/cierre
                    do_sample=True,
                    pad_token_id=gpt2_tokenizer.eos_token_id
                )

            respuesta_gpt2_cruda = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
            texto_generado = respuesta_gpt2_cruda[len(prompt_gpt2):].strip()
            
            # 3. RECONSTRUCCIÓN FINAL FÁCTUAL Y NATURAL
            
            # Usamos la primera parte generada por GPT-2 como introducción
            introduccion = texto_generado.split('\n')[0].split('.')[0] 
            
            # Si GPT-2 no generó una buena introducción, usamos una por defecto
            if len(introduccion) < 10 or 'pelicula' not in introduccion.lower(): 
                introduccion = "Aquí tienes una buena para empezar:"
            else:
                 introduccion += "." # Aseguramos el punto final de la frase

            # Inyección de los datos REALES (el contenido no alucinado)
            respuesta_factual_y_natural = (
                f"{introduccion.strip()}\n"
                f"\"**{info_recuperada['titulo']}**\" ({info_recuperada['anio']})\n"
                f"Género: {info_recuperada['genero']}\n"
                f"{info_recuperada['sinopsis']}"
            )
            
            respuesta_final = respuesta_factual_y_natural

            # Añadir turno del usuario y la respuesta FINAL al historial
            chat_history += f"Usuario: {entrada}\nEnrique: {respuesta_final}\n"

    # --------------------------- LÓGICA GENERATIVA (No RAG) ---------------------------
    else:
        # Añadir turno del usuario al historial
        chat_history += f"Usuario: {entrada}\nEnrique:"

        # Recortar historial
        max_tokens = 700 
        tokens_hist = gpt2_tokenizer.encode(chat_history)
        if len(tokens_hist) > max_tokens:
            tokens_hist = tokens_hist[-max_tokens:]
            chat_history = gpt2_tokenizer.decode(tokens_hist)

        # Crear entrada para GPT-2 y generar (temperatura más alta para creatividad)
        gpt_input = gpt2_tokenizer(chat_history, return_tensors="pt")
        with torch.no_grad():
            output = gpt2_model.generate(
                **gpt_input,
                max_length=len(gpt_input["input_ids"][0]) + 500,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=gpt2_tokenizer.eos_token_id
            )

        respuesta = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        respuesta_final = procesar_respuesta(respuesta, chat_history)

        # Añadir respuesta del bot al historial
        chat_history += f" {respuesta_final}\n"

    # --- Mostrar en la interfaz ---
    output_box.configure(state="normal")
    output_box.insert("end", f"🧑 Tú: {entrada}\n")
    output_box.insert("end", f"🤖 Enrique: {respuesta_final}\n\n")
    output_box.configure(state="disabled")
    input_entry.delete(0, "end")


# --- INTERFAZ GRÁFICA (sin cambios) ---
ctk.set_appearance_mode("light") 
ctk.set_default_color_theme("blue") 

app = ctk.CTk()
app.geometry("700x600")
app.title("Enrique - IA Asistente (RAG Sutil)")
app.configure(fg_color="orange") 

output_box = ctk.CTkTextbox(app, width=650, height=400, font=("Arial", 12), fg_color="salmon")
output_box.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
output_box.configure(state="disabled")

input_entry = ctk.CTkEntry(app, width=500, placeholder_text="Escribe tu mensaje aquí...", text_color="black")
input_entry.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

send_button = ctk.CTkButton(app, text="Enviar", command=responder, fg_color="black")
send_button.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="e")

app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(0, weight=1)

app.mainloop()
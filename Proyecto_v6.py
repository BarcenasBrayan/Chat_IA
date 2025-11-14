import customtkinter as ctk
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import pandas as pd
import numpy as np
import os

# --- 0. CONFIGURACIÓN DE TOKENS ESPECIALES ---
CUSTOM_STOP_TOKEN = "<FIN_SINOPSIS>"

# --- 1. CARGA GLOBAL DE MODELOS ---

# Carga de Modelos base (BETO y GPT-2 para intenciones generales y la introducción)
# NOTA: Ajusta la ruta a tus modelos 'beto_finetuned' y 'gpt2-finetuned'
try:
    beto_model = BertForSequenceClassification.from_pretrained("./beto_finetuned")
    beto_tokenizer = BertTokenizer.from_pretrained("./beto_finetuned")
    beto_model.eval()

    gpt2_model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")
    gpt2_model.eval()
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
except Exception as e:
    print(f"ERROR: No se pudieron cargar los modelos base. {e}")
    class DummyTokenizer:
        def __init__(self): self.eos_token_id = 50256
        def __call__(self, x, **kwargs): return {"input_ids": [[]]}
        def decode(self, x, **kwargs): return ""
    class DummyModel:
        def eval(self): pass
        def generate(self, **kwargs): return [[50256]]
    beto_tokenizer, gpt2_tokenizer = DummyTokenizer(), DummyTokenizer()
    beto_model, gpt2_model = DummyModel(), DummyModel()


# Carga del Modelo de Re-estilización de Sinopsis (gpt2-reestilizado)
try:
    GPT2_SIN_TOKENIZER = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")
    GPT2_SIN_MODEL = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
    
    # *** PASO CLAVE: Añadir el token de parada y obtener su ID ***
    GPT2_SIN_TOKENIZER.add_tokens([CUSTOM_STOP_TOKEN]) 
    
    GPT2_SIN_MODEL.eval()
    GPT2_SIN_TOKENIZER.pad_token = GPT2_SIN_TOKENIZER.eos_token
    print("Modelo de Re-estilización de Sinopsis y token de parada cargados.")
except Exception as e:
    print(f"ADVERTENCIA: No se pudo cargar el modelo reestilizado. Usando la sinopsis original recortada. Error: {e}")
    GPT2_SIN_MODEL = None


# --- 2. CONFIGURACIÓN DE DATOS RAG ---

try:
    # NOTA: Ajusta la ruta a tu archivo CSV
    df_peliculas = pd.read_csv("C:\\Users\\Brayan\\Desktop\\Python\\DatasetPruebas\\peliculas_populares_tmdb.csv")
    
    # Manejar NaNs (Asegúrate de que TONO y TEMA existan en tu CSV)
    df_peliculas = df_peliculas.fillna({
        'Año': np.nan, 'Género': '', 'Reseña': 'Sinopsis no disponible.', 
        'TONO': 'Neutral', 'TEMA': 'General'
    })
    
    df_peliculas['Año_Display'] = df_peliculas['Año'].apply(
        lambda x: str(int(x)) if pd.notna(x) and x > 1900 else '????'
    )
    if 'Puntuacion_Calidad' not in df_peliculas.columns:
         df_peliculas['Puntuacion_Calidad'] = np.random.rand(len(df_peliculas) * 5 + 5)
    
    df_peliculas['Género_Lista'] = df_peliculas['Género'].apply(
        lambda x: [g.strip().lower() for g in str(x).split(',')]
    )
    print("Base de datos de películas cargada.")
except:
    df_peliculas = pd.DataFrame()


chat_history = ""
id2label = {
    0: "info_clave_texto", 1: "recomendacion_peliculas", 2: "sobre_bot", 3: "saludo"
}

def procesar_respuesta(respuesta, prompt, max_chars=800):
    """Limpia la respuesta de GPT-2 para intenciones NO RAG."""
    texto = respuesta[len(prompt):].strip()
    texto = re.sub(r'(Usuario:|<\|endoftext\|>)', '', texto, flags=re.IGNORECASE).strip()
    if "Usuario:" in texto:
        texto = texto.split("Usuario:")[0].strip()
    if "Enrique:" in texto[1:]:
        texto = texto.split("Enrique:")[0].strip()
    if len(texto) > max_chars:
        texto = texto[:max_chars].rsplit(" ", 1)[0] + "..."
    return texto


# --- 3. LÓGICA DE GENERACIÓN Y LIMPIEZA DE SINOPSIS (La Solución) ---

def generar_sinopsis_corta(info_recuperada):
    """
    Genera la sinopsis corta y estilizada usando el token de parada <FIN_SINOPSIS>.
    """
    global GPT2_SIN_MODEL, GPT2_SIN_TOKENIZER 
    
    # Fallback si el modelo re-estilizado no cargó
    if GPT2_SIN_MODEL is None:
        return info_recuperada["sinopsis"][:200].rsplit(' ', 1)[0] + '...' 

    # 1. Obtener el ID del token de parada
    stop_token_id = GPT2_SIN_TOKENIZER.convert_tokens_to_ids(CUSTOM_STOP_TOKEN)
    if stop_token_id == GPT2_SIN_TOKENIZER.unk_token_id:
        # Fallback si el token no se entrenó bien, volvemos al eos genérico para la parada
        stop_token_id = GPT2_SIN_TOKENIZER.eos_token_id

        
    # 2. Construcción del prompt estricto 
    prompt_reestilizado = (
        f"TÍTULO: {info_recuperada['titulo']}\n"
        f"GÉNERO: {info_recuperada['genero']} TONO: {info_recuperada.get('tono', 'Neutral')} TEMA: {info_recuperada.get('tema', 'General')}\n"
        f"SINOPSIS_ORIGINAL: {info_recuperada['sinopsis']}\n"
        f"SINOPSIS_CORTA: " # Aquí inicia la generación
    )
    
    # 3. Generación
    gpt_input = GPT2_SIN_TOKENIZER(prompt_reestilizado, return_tensors="pt")
    
    with torch.no_grad():
        output = GPT2_SIN_MODEL.generate(
            **gpt_input,
            max_length=len(gpt_input["input_ids"][0]) + 40, 
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=GPT2_SIN_TOKENIZER.eos_token_id,
            # *** CLAVE: Usar el token de parada forzada ***
            eos_token_id=stop_token_id, 
        )

    # 4. Decodificación y Limpieza Agresiva
    respuesta_cruda = GPT2_SIN_TOKENIZER.decode(output[0], skip_special_tokens=False)
    
    # 4.1. Extraer SOLO el texto después del marcador SINOPSIS_CORTA:
    try:
        sinopsis_corta = respuesta_cruda.split("SINOPSIS_CORTA:")[1].strip()
    except IndexError:
        sinopsis_corta = respuesta_cruda.strip()

    # 4.2. ELIMINAR EL TOKEN DE PARADA (<FIN_SINOPSIS>) si el modelo lo generó
    if CUSTOM_STOP_TOKEN in sinopsis_corta:
        sinopsis_corta = sinopsis_corta.split(CUSTOM_STOP_TOKEN)[0].strip()

    # 4.3. **CORRECCIÓN DE REPETICIÓN**: Limpieza de tags de entrenamiento residuales (Título, Género, etc.) y bucles.
    sinopsis_corta = re.sub(r'TÍTULO:.*|GÉNERO:.*|SINOPSIS_ORIGINAL:.*|TONO:.*|TEMA:.*', '', sinopsis_corta, flags=re.IGNORECASE).strip()
    if "SINOPSIS_CORTA:" in sinopsis_corta: # Limpia si aún hay un bucle
        sinopsis_corta = sinopsis_corta.split("SINOPSIS_CORTA:")[0].strip()
    
    sinopsis_corta = re.sub(r'[\r\n]+', ' ', sinopsis_corta).strip()
    
    # 4.4. Asegurar que no sea demasiado larga
    if len(sinopsis_corta) > 200:
        sinopsis_corta = sinopsis_corta[:200].rsplit(' ', 1)[0] + '...'
        
    return sinopsis_corta

def obtener_prompt_enriquecido(entrada, chat_history):
    # ... (Lógica para buscar la película y obtener TONO/TEMA del CSV) ...
    global df_peliculas
    
    if df_peliculas.empty: return None, None 

    entrada_lower = entrada.lower()
    genero_buscado = None
    generos_disponibles = set(g for sublist in df_peliculas['Género_Lista'] for g in sublist)
    
    for gen in generos_disponibles:
        if gen in entrada_lower:
            genero_buscado = gen
            break

    candidatas = df_peliculas

    if genero_buscado:
        candidatas = df_peliculas[df_peliculas['Género_Lista'].apply(lambda x: genero_buscado in x)]
    
    if candidatas.empty:
        top_n = max(5, int(len(df_peliculas) * 0.3))
        candidatas = df_peliculas.sort_values(by='Puntuacion_Calidad', ascending=False).head(top_n)

    if candidatas.empty: return None, None
        
    pelicula_seleccionada = candidatas.sample(1).iloc[0]
    
    # Recuperación de datos
    titulo = pelicula_seleccionada.get('Película', 'Título Desconocido').replace('"', '').strip()
    anio = pelicula_seleccionada.get('Año_Display', '????').strip()
    genero = pelicula_seleccionada.get('Género', 'Sin Género').strip()
    reseña_raw = pelicula_seleccionada.get('Reseña', 'Sinopsis no disponible.').strip()
    reseña_limpia = re.sub(r'[\r\n]+', ' ', reseña_raw).strip()
    reseña_limpia = re.sub(r' +', ' ', reseña_limpia)
    tono = pelicula_seleccionada.get('TONO', 'Neutral').strip()
    tema = pelicula_seleccionada.get('TEMA', 'General').strip()

    # CONSTRUCCIÓN DEL PROMPT (Para la INTRODUCCIÓN)
    contexto_factual = (
        f"Datos_Película: Título=\"{titulo}\" Año={anio} Géneros=\"{genero}\" Tono=\"{tono}\" Tema=\"{tema}\""
    )
    
    prompt_gpt2 = (
        f"{chat_history}"
        f"Usuario: {entrada}\n"
        f"{contexto_factual}\n"
        f"Enrique: "
    )
    
    # Información Factual completa para usar en las dos generaciones
    info_recuperada = {
        "titulo": titulo, "anio": anio, "genero": genero, "sinopsis": reseña_limpia,
        "tono": tono, "tema": tema
    }
    
    return prompt_gpt2, info_recuperada


# --- 4. FUNCIÓN DE RESPUESTA PRINCIPAL (CONTROL DE FLUJO) ---

def responder():
    global chat_history

    entrada = input_entry.get().strip()
    if not entrada: return

    # Clasificación de intención (BETO)
    tokens = beto_tokenizer(entrada, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        salida = beto_model(**tokens)
        pred = torch.argmax(salida.logits, dim=1).item()
        intencion = id2label[pred]
    
    respuesta_final = ""
    
    # Recorte de historial
    max_tokens = 700 
    tokens_hist = gpt2_tokenizer.encode(chat_history)
    if len(tokens_hist) > max_tokens:
        tokens_hist = tokens_hist[-max_tokens:]
        chat_history = gpt2_tokenizer.decode(tokens_hist)
    
    
    # LÓGICA RAG SUTIL CONDENSADA
    if intencion == "recomendacion_peliculas":
        
        prompt_gpt2_intro, info_recuperada = obtener_prompt_enriquecido(entrada, chat_history)

        if prompt_gpt2_intro is None:
             respuesta_final = "Lo siento, mi base de datos de películas no está disponible o no hay coincidencias."
        else:
            # Generación 1: Introducción (fluidez)
            gpt_input_intro = gpt2_tokenizer(prompt_gpt2_intro, return_tensors="pt")
            
            with torch.no_grad():
                output_intro = gpt2_model.generate(
                    **gpt_input_intro,
                    temperature=0.6, top_p=0.9, max_length=len(gpt_input_intro["input_ids"][0]) + 20, 
                    do_sample=True, pad_token_id=gpt2_tokenizer.eos_token_id
                )

            respuesta_gpt2_cruda_intro = gpt2_tokenizer.decode(output_intro[0], skip_special_tokens=True)
            texto_generado_intro = respuesta_gpt2_cruda_intro[len(prompt_gpt2_intro):].strip()
            
            introduccion = texto_generado_intro.split('\n')[0].split('.')[0] 
            if len(introduccion) < 10 or 'pelicula' not in introduccion.lower(): 
                introduccion = "Aquí tienes una buena para empezar:"
            else:
                 introduccion += "."

            # Generación 2: Sinopsis Estilizada (precisión y estilo)
            sinopsis_estilizada = generar_sinopsis_corta(info_recuperada) 

            # ENSAMBLAJE FINAL LIMPIO: SOLO SE MUESTRA LA INFO AL USUARIO
            respuesta_factual_y_natural = (
                f"{introduccion.strip()}\n"
                f"\"**{info_recuperada['titulo']}**\" ({info_recuperada['anio']})\n"
                f"Género: {info_recuperada['genero']}\n"
                f"{sinopsis_estilizada}" 
            )
            
            respuesta_final = respuesta_factual_y_natural
            chat_history += f"Usuario: {entrada}\nEnrique: {respuesta_final}\n"

    # LÓGICA GENERATIVA (No RAG)
    else:
        chat_history += f"Usuario: {entrada}\nEnrique:"
        gpt_input = gpt2_tokenizer(chat_history, return_tensors="pt")
        with torch.no_grad():
            output = gpt2_model.generate(
                **gpt_input,
                max_length=len(gpt_input["input_ids"][0]) + 500,
                do_sample=True, top_p=0.95, temperature=0.7, pad_token_id=gpt2_tokenizer.eos_token_id
            )

        respuesta = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        respuesta_final = procesar_respuesta(respuesta, chat_history)
        chat_history += f" {respuesta_final}\n"

    # --- Mostrar en la interfaz ---
    output_box.configure(state="normal")
    output_box.insert("end", f"🧑 Tú: {entrada}\n")
    output_box.insert("end", f"🤖 Enrique: {respuesta_final}\n\n")
    output_box.configure(state="disabled")
    input_entry.delete(0, "end")


# --- 5. INTERFAZ GRÁFICA (sin cambios) ---
ctk.set_appearance_mode("light") 
ctk.set_default_color_theme("blue") 

app = ctk.CTk()
app.geometry("700x600")
app.title("Enrique - IA Asistente (RAG Final)")
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
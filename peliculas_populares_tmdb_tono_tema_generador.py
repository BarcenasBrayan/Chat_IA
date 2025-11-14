import pandas as pd

# === 1. Cargar el archivo original ===
df = pd.read_csv("C:\\Users\\Brayan\\Desktop\\Python\\DatasetPruebas\\peliculas_populares_tmdb.csv")

# === 2. Función para inferir tono y tema automáticamente ===
def infer_tono_y_tema(sinopsis):
    """Genera tono y tema automáticamente a partir de la reseña."""
    if pd.isna(sinopsis) or len(str(sinopsis).strip()) == 0:
        return ("neutro", "indefinido")

    text = sinopsis.lower()

    # --- Determinar tono ---
    if any(w in text for w in ["terror", "muerte", "oscuro", "sangre", "asesino", "misterio"]):
        tono = "oscuro y tenso"
    elif any(w in text for w in ["comedia", "divertido", "risa", "familiar", "aventura", "amistad"]):
        tono = "alegre y aventurero"
    elif any(w in text for w in ["drama", "emocional", "romance", "amor", "tragedia", "vida"]):
        tono = "emocional y dramático"
    elif any(w in text for w in ["acción", "batalla", "guerra", "lucha", "misión"]):
        tono = "intenso y dinámico"
    elif any(w in text for w in ["fantasía", "mágico", "hechizo", "reino", "princesa"]):
        tono = "mágico y épico"
    else:
        tono = "neutral"

    # --- Determinar tema ---
    if any(w in text for w in ["familia", "hermano", "niño", "amistad", "amor"]):
        tema = "familia y relaciones"
    elif any(w in text for w in ["venganza", "justicia", "crimen", "asesino"]):
        tema = "justicia y redención"
    elif any(w in text for w in ["aventura", "descubrir", "viaje", "explorar"]):
        tema = "exploración y descubrimiento"
    elif any(w in text for w in ["misterio", "secreto", "investigación", "asesinato"]):
        tema = "misterio y verdad oculta"
    elif any(w in text for w in ["futuro", "robot", "inteligencia artificial", "tecnología"]):
        tema = "tecnología y humanidad"
    else:
        tema = "superación y crecimiento personal"

    return tono, tema

# === 3. Aplicar la función a todas las reseñas ===
df["Tono"], df["Tema"] = zip(*df["Reseña"].map(infer_tono_y_tema))

# === 4. Guardar el nuevo archivo ===
df.to_csv("peliculas_populares_tmdb_tono_tema.csv", index=False)

print("✅ Archivo generado: peliculas_populares_tmdb_tono_tema.csv")
print(df[["Película", "Tono", "Tema"]].head(10))

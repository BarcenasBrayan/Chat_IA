import pandas as pd
import textwrap

# === 1. Cargar el archivo CSV con tono y tema ===
df = pd.read_csv("peliculas_populares_tmdb_tono_tema.csv")

# === 2. Función para condensar la sinopsis ===
def condensar_sinopsis(texto):
    """
    Resume una sinopsis en aproximadamente dos líneas (150-200 caracteres).
    Se centra en el conflicto o propósito principal.
    """
    if pd.isna(texto) or not texto.strip():
        return "Sinopsis no disponible."

    # Eliminar saltos de línea y reducir texto
    texto = " ".join(texto.split())
    resumen = textwrap.shorten(texto, width=200, placeholder="...")

    return resumen

# === 3. Generar el contenido del dataset ===
salida = []

for _, row in df.iterrows():
    titulo = str(row["Película"]).strip()
    genero = str(row["Género"]).strip()
    tono = str(row["Tono"]).strip()
    tema = str(row["Tema"]).strip()
    sinopsis = str(row["Reseña"]).strip()
    sinopsis_corta = condensar_sinopsis(sinopsis)

    bloque = f"""<|startoftext|>
TÍTULO: {titulo}
GÉNERO: {genero} TONO: {tono} TEMA: {tema}
SINOPSIS_ORIGINAL: {sinopsis}
SINOPSIS_CORTA: {sinopsis_corta} <FIN_SINOPSIS>
<|endoftext|>
"""
    salida.append(bloque)

# === 4. Guardar el archivo de texto ===
with open("dataset_sinopsis_estilizadas.txt", "w", encoding="utf-8") as f:
    f.writelines(salida)

print("✅ Archivo generado: dataset_sinopsis_estilizadas.txt")
print("Ejemplo de salida:\n")
print(salida[0])

import os
import librosa
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pydub import AudioSegment
from collections import Counter

# Directorio de audios
AUDIOS_FOLDER = "/home/sebastian/Desarrollo/Trabajo/shazam/src/audios"


# Extracción de características del audio
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)  # Cargar el archivo de audio
    features = librosa.feature.mfcc(y=y, sr=sr)  # Extraer características MFCC

    if len(features) == 0:
        raise ValueError("No se pudieron extraer características del archivo de audio.")

    return features


# Generación de huellas dactilares de audio
def generate_fingerprints(audio_file):
    audio = AudioSegment.from_file(audio_file, format="mp3")
    duration = len(audio) / 1000.0  # Duración del audio en segundos
    audio_fingerprints = []

    # Dividir el audio en fragmentos de 5 segundos
    fragment_duration = 5
    start_time = 0

    while start_time + fragment_duration <= duration:
        end_time = start_time + fragment_duration
        audio_fragment = audio[
            start_time * 1000 : end_time * 1000
        ]  # Extraer el fragmento de audio
        temp_file = "/tmp/fragment.wav"  # Ruta del archivo WAV temporal

        # Guardar el fragmento de audio en formato WAV temporal
        audio_fragment.export(temp_file, format="wav")

        # Extraer características del fragmento de audio
        features = extract_features(temp_file)

        # Generar una huella dactilar única para el fragmento de audio
        fingerprint = np.asarray(features.flatten())

        # Agregar la huella dactilar a la lista
        audio_fingerprints.append(fingerprint)

        # Actualizar el tiempo de inicio para el siguiente fragmento
        start_time += fragment_duration

        # Eliminar el archivo WAV temporal
        os.remove(temp_file)

    return audio_fingerprints


# Construcción de un índice de búsqueda
def build_index(audios_folder):
    fingerprints_list = []
    labels_list = []

    # Recorrer los archivos de audio en la carpeta
    for filename in os.listdir(audios_folder):
        audio_file = os.path.join(audios_folder, filename)

        if audio_file.endswith(".mp3"):
            fingerprints = generate_fingerprints(audio_file)
            fingerprints_list.extend(fingerprints)
            labels_list.extend(
                [filename[:-4]] * len(fingerprints)
            )  # Eliminar la extensión del archivo

    if not fingerprints_list:
        raise ValueError("No se encontraron archivos de audio válidos en la carpeta.")

    fingerprints_array = np.array(fingerprints_list)
    index = NearestNeighbors(n_neighbors=1).fit(fingerprints_array)

    return index, labels_list


# Generación de huellas dactilares del fragmento de audio
def generate_query_fingerprints(audio_file):
    return generate_fingerprints(audio_file)


# Búsqueda y comparación de huellas dactilares
def identify_song(audio_file, index, labels_list):
    query_fingerprints = generate_query_fingerprints(audio_file)
    query_fingerprints_array = np.array(query_fingerprints).reshape(1, -1)
    distances, indices = index.kneighbors(query_fingerprints_array)

    if distances.size > 0:
        min_distance_index = np.argmin(distances.reshape(1, -1))
        song_label = labels_list[indices[min_distance_index][0]]
        return song_label
    else:
        return "Canción no encontrada"


# Identificación de la canción
def main():
    # Validar la disponibilidad de la carpeta de audios
    if not os.path.isdir(AUDIOS_FOLDER):
        print("Error: No se encontró la carpeta de audios:", AUDIOS_FOLDER)
        exit(1)

    try:
        # Construir el índice de búsqueda
        index, labels_list = build_index(AUDIOS_FOLDER)

        # Obtener el fragmento de audio a identificar
        audio_file = "/home/sebastian/Desarrollo/Trabajo/shazam/src/test/test6.mp3"

        # Verificar la existencia del archivo de audio de prueba
        if not os.path.isfile(audio_file):
            print("Error: No se encontró el archivo de audio de prueba:", audio_file)
            exit(1)

        # Búsqueda de los vecinos más cercanos
        target_fingerprints = generate_query_fingerprints(audio_file)
        target_fingerprints_array = np.array(target_fingerprints)
        distances, indices = index.kneighbors(target_fingerprints_array)

        # Imprimir el resultado
        if distances.size > 0:
            nearest_neighbors = indices.flatten()
            song_names = [labels_list[i] for i in nearest_neighbors]

            # Contar las ocurrencias de cada canción
            song_counts = Counter(song_names)

            # Obtener la canción con el mayor número de ocurrencias
            most_common_song = song_counts.most_common(1)[0][0]

            print("El nombre de la canción es:", most_common_song)
        else:
            print("No se encontró la canción.")

    except Exception as e:
        print("Error:", str(e))


if __name__ == "__main__":
    main()

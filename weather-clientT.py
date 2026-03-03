import os
import cv2
import numpy as np
import requests
import json

# ==========================
# CONFIGURACIÓN
# ==========================
MODEL_NAME = "weather-model"
SERVER_URL = f"https://weather-model-w6cb.onrender.com/v1/models/{MODEL_NAME}:predict"
DATA_ROOT = "dataset-original"
IMG_SIZE = 224

# ==========================
# FUNCIONES
# ==========================
def mobilenetv2_preprocess_input(x):
    """
    Implementación de tf.keras.applications.mobilenet_v2.preprocess_input sin TF.
    Escala de [0,255] a [-1,1].
    """
    x = x.astype(np.float32)
    return (x / 127.5) - 1.0

def preprocess_image(img_path):
    """Carga y preprocesa la imagen sin preprocess_input para coincidir con el entrenamiento."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Imagen no encontrada o corrupta: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)  # Mantener [0,255] como en el entrenamiento
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    return img.tolist()

# ==========================
# MAIN
# ==========================
def main():
    ## Cargar etiquetas de clases desde el archivo generado por el entrenamiento
    classes_file = os.path.join("weather-model", "classes.txt")
    with open(classes_file, 'r', encoding='utf-8') as f:
        CLASSES_LABELS = [line.strip() for line in f]
    
    print(f"Clases cargadas: {CLASSES_LABELS}")
    print("🔹 Probando una imagen por cada clase de clima...")
    
    for class_name in CLASSES_LABELS:
        folder_path = os.path.join(DATA_ROOT, class_name)
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder no encontrado: {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
        if not files:
            print(f"⚠️ No hay archivos .jpg en {folder_path}")
            continue

        img_path = os.path.join(folder_path, files[0])

        try:
            img = preprocess_image(img_path)
            payload = json.dumps({"instances": img})

            response = requests.post(SERVER_URL, data=payload, timeout=60)  # 60-second timeout
            response.raise_for_status()
            prediction = response.json()["predictions"][0]

            predicted_class = CLASSES_LABELS[np.argmax(prediction)]
            confidence = max(prediction)
            print(f"{class_name}: Predicho como {predicted_class} (confianza: {confidence:.4f})")

        except Exception as e:
            print(f"❌ Error con {class_name}: {e}")

if __name__ == "__main__":
    main()

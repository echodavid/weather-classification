import os
import cv2
import numpy as np
import requests
import json

MODEL_NAME = "weather-model"
SERVER_URL = f"http://localhost:8000/v1/models/{MODEL_NAME}:predict"
CLASSES_LABELS = ["dew", "fogsmog", "frost", "glaze", "hail",
                  "lightning", "rain", "rainbow", "rime",
                  "sandstorm", "snow"]
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
        classes_labels = [line.strip() for line in f]
    
    print(f"Clases cargadas: {classes_labels}")
    print("Testing with one image from each weather class:")
    
    for class_name in classes_labels:
        ## Construir la ruta a la carpeta de la clase
        folder_path = os.path.join(DATA_ROOT, class_name)
        try:
            ## Listar archivos de imagen en la carpeta
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not files:
                print(f"No image files in {folder_path}")
                continue
            ## Tomar la primera imagen
            img_path = os.path.join(folder_path, files[0])
            ## Cargar y preprocesar la imagen
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32)  # Mantener [0,255] como en el entrenamiento
            img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
            img = img.tolist()
            predict_request = json.dumps({'instances': img})

            ## Enviar solicitud al servidor
            response = requests.post(SERVER_URL, data=predict_request, timeout=30)
            response.raise_for_status()
            prediction = response.json()['predictions'][0]
            
            ## Obtener la clase predicha y confianza
            predicted_class = classes_labels[np.argmax(prediction)]
            confidence = max(prediction)
            
            print(f"{class_name}: Predicted as {predicted_class} (confidence: {confidence:.4f})")
        except Exception as e:
            print(f"Error with {class_name}: {e}")

if __name__ == '__main__':
    main()

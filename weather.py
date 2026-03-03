import os
import cv2
import numpy as np
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================
# CONFIGURACIÓN
# ==========================
IMG_SIZE = 224
MODEL_NAME = "weather-mobilenetv2"
EPOCHS = 20
BATCH_SIZE = 16
DATASET_PATH = "dataset-original"

# ==========================
# DESCARGAR DATASET SI NO EXISTE
# ==========================
if not os.path.exists(DATASET_PATH):
    print("🔹 Descargando dataset desde KaggleHub...")
    dataset_path = kagglehub.dataset_download("jehanbhathena/weather-dataset")
    DATA_ROOT = os.path.join(dataset_path, "dataset")
    os.makedirs(DATASET_PATH, exist_ok=True)
    # copiar contenido a dataset-original
    import shutil
    shutil.copytree(DATA_ROOT, DATASET_PATH, dirs_exist_ok=True)
    print("✅ Dataset descargado en:", DATASET_PATH)
else:
    print("✅ Dataset ya existe:", DATASET_PATH)
    DATA_ROOT = DATASET_PATH

# ==========================
# CARGAR IMÁGENES
# ==========================
images, labels = [], []
valid_ext = (".jpg", ".jpeg", ".png")

for class_name in sorted(os.listdir(DATA_ROOT)):
    folder = os.path.join(DATA_ROOT, class_name)
    if not os.path.isdir(folder):
        continue
    print("Leyendo clase:", class_name)
    for fname in os.listdir(folder):
        if not fname.lower().endswith(valid_ext):
            continue
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ Imagen corrupta: {path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img)
        labels.append(class_name)

images = np.array(images, dtype=np.float32)
labels = np.array(labels)
print(f"Total imágenes cargadas: {len(images)}")

# ==========================
# LABEL ENCODING
# ==========================
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)

# ==========================
# TRAIN / TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

# ==========================
# CALCULAR PESOS DE CLASE
# ==========================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_encoded),
    y=labels_encoded
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# ==========================
# DATA AUGMENTATION
# ==========================
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

valid_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# ==========================
# MODELO MobileNetV2
# ==========================
base = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",  # RGB
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# ENTRENAMIENTO
# ==========================
model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=valid_gen.flow(X_test, y_test),
    epochs=EPOCHS,
    class_weight=class_weights
)

# ==========================
# EXPORT PARA TF SERVING
# ==========================
EXPORT_PATH = os.path.join(MODEL_NAME, "1")
os.makedirs(EXPORT_PATH, exist_ok=True)

@tf.function(input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)])
def serving_fn(inputs):
    x = tf.image.resize(inputs, (IMG_SIZE, IMG_SIZE))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    preds = model(x, training=False)
    return {"predictions": preds}

tf.saved_model.save(
    model,
    EXPORT_PATH,
    signatures={"serving_default": serving_fn}
)

print("✅ Modelo exportado para TF Serving en:", EXPORT_PATH)

# ==========================
# GUARDAR CLASES
# ==========================
classes_path = os.path.join(MODEL_NAME, "classes.txt")
with open(classes_path, "w") as f:
    for c in encoder.classes_:
        f.write(c + "\n")
print("✅ Clases guardadas en:", classes_path)

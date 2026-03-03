import os
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer

# ==========================
# CONFIGURACIÓN
# ==========================
MODEL_NAME = "weather-model"  # Nombre del modelo entrenado
EXPORT_PATH = os.path.join(MODEL_NAME, "1")  # Ruta donde está guardado el modelo actual
NEW_EXPORT_PATH = os.path.join(MODEL_NAME + "_fixed", "1")  # Nueva ruta para el modelo corregido

# ==========================
# CREAR TFSMLAYER Y NUEVO MODELO
# ==========================
print("🔄 Creando TFSMLayer y nuevo modelo...")
try:
    # Crear TFSMLayer con la ruta del SavedModel
    tfsm_layer = TFSMLayer(EXPORT_PATH, call_endpoint='serving_default')
    
    # Crear un nuevo modelo Sequential que use el TFSMLayer
    model_fixed = tf.keras.Sequential([
        tfsm_layer
    ])
    
    # Construir el modelo con la forma de entrada
    model_fixed.build(input_shape=(None, 224, 224, 3))
    
    print("✅ Nuevo modelo creado exitosamente")
except Exception as e:
    print(f"❌ Error al crear el nuevo modelo: {e}")
    exit(1)

# ==========================
# GUARDAR MODELO CORREGIDO
# ==========================
print("🔄 Guardando modelo corregido en:", NEW_EXPORT_PATH)
os.makedirs(NEW_EXPORT_PATH, exist_ok=True)
model_fixed.export(NEW_EXPORT_PATH)
print("✅ Modelo guardado correctamente")

# ==========================
# VERIFICAR QUE FUNCIONE
# ==========================
print("🔄 Verificando carga del modelo corregido...")
try:
    model_verify = tf.keras.models.load_model(NEW_EXPORT_PATH)
    print("✅ Modelo corregido cargado exitosamente")
    print("📊 Resumen del modelo:")
    model_verify.summary()
except Exception as e:
    print(f"❌ Error al verificar el modelo corregido: {e}")

print("🎉 Proceso completado. Usa el modelo en:", NEW_EXPORT_PATH)

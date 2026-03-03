import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List

# ==========================
# CONFIGURACIÓN
# ==========================
MODEL_PATH = "weather-model_fixed/1"
CLASSES_LABELS = ["dew", "fogsmog", "frost", "glaze", "hail",
                  "lightning", "rain", "rainbow", "rime",
                  "sandstorm", "snow"]

# Cargar el modelo
try:
    model = tf.saved_model.load(MODEL_PATH)
    infer = model.signatures["serving_default"]
    print("✅ Modelo cargado exitosamente.")
except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")
    model = None

# ==========================
# FASTAPI APP
# ==========================
app = FastAPI(title="Weather Prediction API", description="API para predecir clases de clima usando TensorFlow.")

class PredictRequest(BaseModel):
    instances: List[List[List[List[float]]]]  # (batch, 224, 224, 3)

@app.post("/v1/models/weather-model:predict")
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")

    try:
        print("📥 Recibiendo solicitud de predicción...")
        # Convertir a tensor
        instances = np.array(request.instances, dtype=np.float32)
        print(f"📊 Forma de entrada: {instances.shape}")
        # Crear input tensor
        input_tensor = tf.convert_to_tensor(instances)
        print("🔄 Ejecutando inferencia...")
        # Inferencia
        predictions = infer(input_tensor)
        print(f"🔍 Claves de salida: {list(predictions.keys())}")
        # Obtener las predicciones (asumiendo output key 'predictions')
        pred_key = list(predictions.keys())[0]
        pred_array = predictions[pred_key].numpy()
        print(f"✅ Predicciones generadas: {pred_array.shape}")

        # Convertir a lista y formatear respuesta
        response = {"predictions": pred_array.tolist()}
        print("📤 Enviando respuesta.")
        return response
    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Weather Prediction API is running. Use POST /v1/models/weather-model:predict"}

@app.get("/test")
async def test():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

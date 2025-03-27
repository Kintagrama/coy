from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import io
import os

# Configuración inicial
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Habilita CORS para todas las rutas

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

# Carga del modelo (solo una vez al iniciar)
MODEL_NAME = "tuphamdf/skincare-detection"
print("⏳ Cargando modelo de IA...")
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model.eval()
print("✅ Modelo cargado correctamente")

# Ruta para análisis de imágenes
@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        # 1. Validar que se envió una imagen
        if 'image' not in request.files:
            return jsonify({"error": "No se subió ninguna imagen"}), 400
        
        # 2. Leer la imagen
        image_file = request.files['image']
        
        # 3. Convertir a formato PIL
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        
        # 4. Preprocesar y predecir
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 5. Procesar resultados
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_probs, top_labels = torch.topk(probs, 3)
        
        # 6. Formatear respuesta
        results = [{
            "label": model.config.id2label[top_labels[0][i].item()],
            "score": top_probs[0][i].item(),
            "confidence": f"{(top_probs[0][i].item() * 100):.1f}%"
        } for i in range(3)]
        
        return jsonify({
            "success": True,
            "predictions": results
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error en el análisis: {str(e)}"
        }), 500

# Health check
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

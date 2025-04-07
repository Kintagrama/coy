from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import io

# Configuración de la aplicación Flask
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {"origins": "*"},
    r"/health": {"origins": "*"}
})

# Configuración del modelo
MODEL_NAME = "tuphamdf/skincare-detection"

print("⏳ Cargando modelo de IA...")
try:
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=False,
        device_map=None,
        torch_dtype=torch.float32
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model.eval()
    print("✅ Modelo cargado correctamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {str(e)}")
    print("Intentando carga básica...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model.eval()
    print("✅ Modelo cargado (modo básico)")

# Ruta para servir el frontend


@app.route('/')
def serve_index():
    return send_file('index.html')
    
# Ruta para análisis de imágenes
@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"}), 400
        
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
        
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_probs, top_labels = torch.topk(probs, 3)  # Obtener top 3 resultados
        
        predictions = [{
            "label": model.config.id2label[top_labels[0][i].item()],
            "confidence": f"{(top_probs[0][i].item() * 100):.1f}%"
        } for i in range(3)]
        
        return jsonify({
            "success": True,
            "predictions": predictions
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Health check
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "loaded": model is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

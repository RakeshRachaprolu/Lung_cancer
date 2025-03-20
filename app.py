from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import json
import os

app = Flask(__name__)

# Load the ensemble
def load_ensemble(save_dir="ensemble_model"):
    model_names = ['mobilenet', 'inceptionv3', 'vgg16', 'densenet121']
    loaded_models = [load_model(os.path.join(save_dir, f"{name}.h5")) for name in model_names]
    with open(os.path.join(save_dir, "ensemble_weights.json"), "r") as f:
        loaded_weights = json.load(f)
    with open(os.path.join(save_dir, "class_names.json"), "r") as f:
        loaded_class_names = json.load(f)
    return loaded_models, loaded_weights, loaded_class_names

# Prediction function
def weighted_ensemble_predict(models, weights, img_array):
    predictions = [model.predict(img_array) * weight for model, weight in zip(models, weights.values())]
    weighted_predictions = np.sum(predictions, axis=0)
    return weighted_predictions

# Load models at startup
models, weights, class_names = load_ensemble()
img_size = (224, 224)  # Adjust based on your model's input size

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    try:
        img = load_img(file, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        ensemble_pred = weighted_ensemble_predict(models, weights, img_array)
        pred_class_idx = np.argmax(ensemble_pred, axis=1)[0]
        pred_class = class_names[pred_class_idx]
        confidence = float(ensemble_pred[0][pred_class_idx])

        return jsonify({'prediction': pred_class, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
import os
import gdown
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from utils.preprocess_text import TextPreprocessor

app = Flask(__name__)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class ToxicityModelAPI:
    def __init__(self, model_path, tokenizer_path, max_pad_len):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_pad_len = max_pad_len
        self.text_preprocessor = TextPreprocessor(self.tokenizer_path, self.max_pad_len)
        self.model = self.load_model(self.model_path)

    def load_model(self, model_path):
        os.makedirs("models", exist_ok=True)

        # Download model if not found
        if not os.path.exists(model_path):
            print("Model not found. Downloading...")
            url = "https://drive.google.com/uc?export=download&id=1hGhAbTFSxF-u2_uRrVk0jetp_vsqBw_U"
            gdown.download(url, model_path, quiet=False)
        
        # Verify file exists after download
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path} after download.")

        try:
            print("Loading model...")
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
            return model
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")

    def predict(self, comment_text):
        # Preprocess text
        preprocessed_text = self.text_preprocessor.preprocess(comment_text)
        
        # Model prediction
        predictions = self.model.predict(preprocessed_text)
        
        # Prepare response
        response = {
            'toxic': round(float(predictions[0][0]), 2),
            'severe_toxic': round(float(predictions[0][1]), 2),
            'obscene': round(float(predictions[0][2]), 2),
            'threat': round(float(predictions[0][3]), 2),
            'insult': round(float(predictions[0][4]), 2),
            'identity_hate': round(float(predictions[0][5]), 2)
        }
        return response


# API Initialization
model_path = 'models/comment_tox_model.keras'
tokenizer_path = 'tokenizer/tokenizer.pkl'
max_pad_len = 300

toxicity_api = ToxicityModelAPI(model_path, tokenizer_path, max_pad_len)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        comment = data.get('comment', '')
        if not comment:
            return jsonify({'error': 'No comment provided'}), 400
        
        # Get prediction
        result = toxicity_api.predict(comment)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def home():
    return "Toxicity Detection API is running."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from utils.preprocess_text import TextPreprocessor
import os

app = Flask(__name__)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class ToxicityModelAPI:
    def __init__(self, model_path, tokenizer_path, max_pad_len):
        self.model = self.load_model(model_path)
        self.text_preprocessor = TextPreprocessor(tokenizer_path, max_pad_len)

    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path)
        return model

    def predict(self, comment_text):
        # Preprocess the comment text
        preprocessed_text = self.text_preprocessor.preprocess(comment_text)
        
        # Model prediction
        predictions = self.model.predict(preprocessed_text)
        
        # Assuming the output layer has 6 sigmoid units
        response = {
            'toxic': round(float(predictions[0][0]), 2),
            'severe_toxic': round(float(predictions[0][1]), 2),
            'obscene': round(float(predictions[0][2]), 2),
            'threat': round(float(predictions[0][3]), 2),
            'insult': round(float(predictions[0][4]), 2),
            'identity_hate': round(float(predictions[0][5]), 2)
        }
        return response


# Initialize the API object with paths and parameters
model_path = 'models/comment_tox_model.keras'          # Path to your trained model
tokenizer_path = 'tokenizer/tokenizer.pkl'             # Path to your saved tokenizer
max_pad_len = 300                               # Should match the value used during training

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
    # Ensure the embedding file is loaded by the model
    app.run(host='0.0.0.0', port=5000)

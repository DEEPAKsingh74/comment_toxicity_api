# Load model globally
MODEL_PATH = os.path.join('../models', 'comment_tox_model.keras')

# Load the model once and keep it in memory
model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Preprocessing function (adjust based on your data pipeline)
def preprocess_text(text: str):
	custom_text = clean_text(text)
	custom_text = lemma(custom_text)
	custom_text = remove_stopwords(custom_text)

	custom_tokenized = tokenizer.texts_to_sequences([custom_text])

	custom_padded = pad_sequences(custom_tokenized, maxlen=MAXPADLEN, padding="post")
	return custom_padded
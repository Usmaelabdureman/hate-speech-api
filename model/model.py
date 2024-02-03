from pathlib import Path
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/tokenizer_0_1_0.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open(f"{BASE_DIR}/label_encoder_0_1_0.pkl", "rb") as file:
    le = pickle.load(file)

amharic_model = load_model(f"{BASE_DIR}/detectHate_model-0_1_0.h5")
# amharic_model = load_model(f"{BASE_DIR}/quantized_model.tflite")
# oromo_model = load_model(f"{BASE_DIR}/detectHate_oro_model-0_1_0.h5")

def predict_pipeline(text):
    # Tokenize the text using the tokenizer
    encoded_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(encoded_text, padding='post', maxlen=55)
    predictions = amharic_model.predict(padded_text).argmax(axis=1)
    predicted_label = le.inverse_transform(predictions)[0]
    return predicted_label
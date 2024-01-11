from pathlib import Path
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/tokenizer_0_1_0.pkl", "rb") as file:
    tokenizer = pickle.load(file)

with open(f"{BASE_DIR}/label_encoder_0_1_0.pkl", "rb") as file:
    le = pickle.load(file)

loaded_model = load_model(f"{BASE_DIR}/detectHate_model-0_1_0.h5")

def predict_pipeline(text):
    # Tokenize the text using the tokenizer
    encoded_text = tokenizer.texts_to_sequences([text])

    # Pad the sequence to a fixed length
    padded_text = pad_sequences(encoded_text, padding='post', maxlen=55)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(padded_text).argmax(axis=1)

    # Convert predictions back to labels using the label encoder
    predicted_label = le.inverse_transform(predictions)[0]
    return predicted_label
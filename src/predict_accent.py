from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import numpy as np
import pandas as pd
import joblib
from pydub import AudioSegment
import os
from keras import models

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
pre_trained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def convert_wav(file_path: str) -> str | None:
    try:
        sound = AudioSegment.from_mp3(file_path)
        file_path = file_path.split(".mp3")[0] + ".wav"
        sound.export(f"{file_path}", format="wav")
        return file_path
    except Exception as e:
        print(e)
        return None

def extract_audio_features(file_path: str) -> pd.DataFrame | None:
    try:
        if '.mp3' in file_path:
            file_path = convert_wav(file_path)
            mp3_file_path = file_path.split(".wav")[0] + ".mp3"
            os.remove(mp3_file_path)

        audio, sample_rate = librosa.load(file_path, sr=16000)  # Match Wav2Vec2's 16kHz sample rate

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = pre_trained_model(**inputs)
            embeddings = outputs.last_hidden_state

        feature_vector = torch.mean(embeddings, dim=1).squeeze().numpy()

        feature_names = [f'feature_{i+1}' for i in range(feature_vector.shape[0])]
        return pd.DataFrame([feature_vector], columns=feature_names)
    
    except Exception as e:
        print(e)
        return None
    
def predict_audio_accent(file_path: str) -> str | None:
    try:
        model = models.load_model('model/accent_predictor_model.keras')
        scaler = joblib.load('model/scaler.pkl')
        label_encoder = joblib.load('model/label_encoder.pkl')

        new_audio_features = extract_audio_features(file_path)

        new_audio_features_scaled = scaler.transform(new_audio_features)

        predicted_class_probabilities = model.predict(new_audio_features_scaled)
        predicted_class_index = np.argmax(predicted_class_probabilities, axis=1)
        predicted_accent = label_encoder.inverse_transform(predicted_class_index)
        
        prediction_confidence = np.max(predicted_class_probabilities) * 100

        # print(f"The predicted accent is: {predicted_accent[0].capitalize()}")
        # print(f"Prediction confidence: {prediction_confidence:.2f}%")

        return predicted_accent[0].capitalize(), prediction_confidence
    
    except Exception as e:
        print(e)
        return None, None

if __name__ == '__main__':
    file_path = 'input_files/nz.wav'
    predicted_accent, predicted_precent = predict_audio_accent(file_path)

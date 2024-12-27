from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from train_model import train_audio_model
from tqdm import tqdm

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
pre_trained_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def extract_features(directory: str) -> pd.DataFrame | None:
    try:
        features = []
        for accent in tqdm(os.listdir(directory), desc=f"Extraction started"):
            accent_path = os.path.join(directory, accent)

            for audio_file in os.listdir(accent_path):
                if audio_file.endswith('.wav') or audio_file.endswith('.mp3'):
                    file_path = os.path.join(accent_path, audio_file)
                    try:
                        audio, sample_rate = librosa.load(file_path, sr=16000)

                        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

                        with torch.no_grad():
                            outputs = pre_trained_model(**inputs)
                            embeddings = outputs.last_hidden_state

                        feature_vector = torch.mean(embeddings, dim=1).squeeze().numpy()

                        temp_features = {"file": audio_file, "accent": accent.replace("_", " ")}
                        for index, value in enumerate(feature_vector):
                            temp_features[f'feature_{index + 1}'] = value

                        features.append(temp_features)
                    
                    except Exception as audio_error:
                        print(f"Error processing {file_path}: {audio_error}")

        return pd.DataFrame(features)
    
    except Exception as e:
        print(e)
        return None

if __name__ == '__main__':
    feature_df = extract_features('data/audio_samples')
    feature_df = shuffle(feature_df)
    feature_df.to_csv("data/csv_files/sample_audio_features.csv", index=False)
    train_audio_model()

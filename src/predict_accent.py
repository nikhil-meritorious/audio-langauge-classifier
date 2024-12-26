import librosa
import numpy as np
import pandas as pd
import joblib
from pydub import AudioSegment
import os
from keras import models

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

        audio, sample_rate = librosa.load(file_path, sr=22050)

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfccs.T, axis=0)

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0])
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio)[0])

        spectral_contrast_mean = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_bands=5), axis=1)
        chroma_mean = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate), axis=1)

        rms = np.mean(librosa.feature.rms(y=audio)[0])
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))

        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128), axis=1)

        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate), axis=1)

        features = [spectral_centroid, zero_crossing_rate, rms, spectral_bandwidth] + list(mfcc_mean) + list(chroma_mean) + list(spectral_contrast_mean) + list(mel_spectrogram) + list(tonnetz)

        feature_names = ['spectral_centroid', 'zcr', 'rms', 'spectral_bandwidth'] + \
                        [f'mfcc{i+1}' for i in range(len(mfcc_mean))] + \
                        [f'chroma{i+1}' for i in range(len(chroma_mean))] + \
                        [f'spectral_contrast{i+1}' for i in range(len(spectral_contrast_mean))] + \
                        [f'mel{i+1}' for i in range(len(mel_spectrogram))] + \
                        [f'tonnetz{i+1}' for i in range(len(tonnetz))]

        return pd.DataFrame([features], columns=feature_names)
    
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

        predicted_class = model.predict(new_audio_features_scaled)
        predicted_accent = label_encoder.inverse_transform(np.argmax(predicted_class, axis=1))

        print(f"The predicted accent is: {predicted_accent[0].capitalize()}")
        return predicted_accent[0].capitalize()
    
    except Exception as e:
        print(e)
        return None

if __name__ == '__main__':
    file_path = 'input_files/nz.wav'
    predicted_accent = predict_audio_accent(file_path)

    print(f"The predicted accent is: {predicted_accent}")

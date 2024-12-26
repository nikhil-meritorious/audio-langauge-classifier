import librosa
import numpy as np
import pandas as pd
import os
from sklearn.utils import shuffle
from train_model import train_audio_model
from tqdm import tqdm

def extract_features(directory: str) -> pd.DataFrame | None:
    try:
        features = []
        for accent in tqdm(os.listdir(directory)):
            print(f'extracting {accent}')
            for audio_file in os.listdir(f'{directory}/{accent}'):
                if audio_file.endswith('.wav') or audio_file.endswith('.mp3'):
                    file_path = os.path.join(f'{directory}/{accent}', audio_file)

                    audio, sample_rate = librosa.load(file_path, sr=22050)

                    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                    mfcc_mean = np.mean(mfccs.T, axis=0)

                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0])
                    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
                    
                    spectral_contrast_mean = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_bands=5), axis=1)

                    chroma_mean = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate), axis=1)

                    rms = np.mean(librosa.feature.rms(y=audio)[0])
                    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate))

                    temp_features = {
                        "file": audio_file,
                        "accent": accent.replace("_", " "),
                        "spectral_centroid": spectral_centroid,
                        "zcr": zero_crossing_rate,
                        "rms": rms,
                        "spectral_bandwidth": spectral_bandwidth
                    }

                        
                    for index, mfcc in enumerate(mfcc_mean):
                        temp_features[f'mfcc{index + 1}'] = mfcc

                    for index, chroma in enumerate(chroma_mean):
                        temp_features[f'chroma{index + 1}'] = chroma

                    for index, spectral_contrast in enumerate(spectral_contrast_mean):
                        temp_features[f'spectral_contrast{index + 1}'] = spectral_contrast
                    
                    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128), axis=1)
                    
                    for index, mel in enumerate(mel_spectrogram):
                        temp_features[f'mel{index + 1}'] = mel

                    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate), axis=1)
                    
                    for index, tonnetz_value in enumerate(tonnetz):
                        temp_features[f'tonnetz{index + 1}'] = tonnetz_value

                    features.append(temp_features)

        return pd.DataFrame(features)
            
    except Exception as e:
        print(e)
        return None

if __name__ == '__main__':
    feature_df = extract_features('data/audio_samples')
    feature_df = shuffle(feature_df)
    feature_df.to_csv("data/csv_files/sample_audio_features.csv", index=False)
    train_audio_model()

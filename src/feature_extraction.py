import librosa
import numpy as np
import pandas as pd
import os

def extract_features(directory):
    try:
        features = []
        for audio_file in os.listdir(directory):
            if audio_file.endswith('.wav') or audio_file.endswith('.mp3'):
                label = audio_file.split(".")[0]
                file_path = os.path.join(directory, audio_file)

                audio, sample_rate = librosa.load(file_path, sr=None)

                mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
                mfcc_mean = np.mean(mfccs.T, axis=0)

                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0])
                zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
                
                spectral_contrast_mean = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_bands=5), axis=1)

                chroma_mean = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate), axis=1)

                temp_features = {
                    "file": audio_file,
                    "accent": label,
                    "spectral_centroid": spectral_centroid,
                    "zcr": zero_crossing_rate
                }

                for index, mfcc in enumerate(mfcc_mean):
                    temp_features[f'mfcc{index + 1}'] = mfcc

                for index, chroma in enumerate(chroma_mean):
                    temp_features[f'chroma{index + 1}'] = chroma

                for index, spectral_contrast in enumerate(spectral_contrast_mean):
                    temp_features[f'spectral_contrast{index + 1}'] = spectral_contrast

                features.append(temp_features)

        return pd.DataFrame(features)
            
    except Exception as e:
        print(e)
        return None

if __name__ == '__main__':
    feature_df = extract_features('data/audio_samples')
    feature_df.to_csv("data/csv_files/sample_audio_features.csv", index=False)

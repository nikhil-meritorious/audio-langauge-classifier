import librosa
import numpy as np
import joblib

def extract_audio_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfccs.T, axis=0)

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0])
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio)[0])

    spectral_contrast_mean = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_bands=5), axis=1)
    chroma_mean = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate), axis=1)

    features = [spectral_centroid, zero_crossing_rate] + list(mfcc_mean) + list(chroma_mean) + list(spectral_contrast_mean)

    return np.array(features).reshape(1, -1)

if __name__ == '__main__':  
    model = joblib.load('model/accent_predictor_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')

    file_path = 'luvvoice.com-20241213-TQtdLS.mp3'
    new_audio_features = extract_audio_features(file_path)

    new_audio_features_scaled = scaler.transform(new_audio_features)

    predicted_class = model.predict(new_audio_features_scaled)
    predicted_accent = label_encoder.inverse_transform(predicted_class)

    print(f"The predicted accent/language is: {predicted_accent[0].capitalize()}")

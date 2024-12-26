import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_audio_model() -> None:
    try:
        feature_df = pd.read_csv('data/csv_files/sample_audio_features.csv')

        X = feature_df.drop(columns=['file', 'accent'])
        y = feature_df['accent']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model accuracy: {accuracy * 100:.2f}%')

        joblib.dump(model, 'model/accent_predictor_model.pkl')
        joblib.dump(scaler, 'model/scaler.pkl')
        joblib.dump(label_encoder, 'model/label_encoder.pkl')

    except Exception as e:
        print(e)
        return None

if __name__ == '__main__':
    train_audio_model()

from keras import models, layers, optimizers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

def train_audio_model() -> None:
    try:
        feature_df = pd.read_csv('data/csv_files/sample_audio_features.csv')

        X = feature_df.drop(columns=['file', 'accent'])
        y = feature_df['accent']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = models.Sequential([
            layers.Dense(512, input_dim=X_train_scaled.shape[1], activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(label_encoder.classes_), activation='softmax')
        ])


        model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=2)

        loss, accuracy = model.evaluate(X_test_scaled, y_test)
        print(f'Model accuracy: {accuracy * 100:.2f}%')

        model.save('model/accent_predictor_model.keras')
        joblib.dump(scaler, 'model/scaler.pkl')
        joblib.dump(label_encoder, 'model/label_encoder.pkl')

    except Exception as e:
        print(e)
        return None

if __name__ == '__main__':
    train_audio_model()

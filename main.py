from fastapi import FastAPI, UploadFile, status, File
from fastapi.responses import JSONResponse
from src.predict_accent import predict_audio_accent
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

valid_extenstions = ["mp3", "wav"]
upload_directory = "input_files"

@app.get("/")
def root():
    return {"message": "Welcome to the audio accent classification api"}

@app.post("/api/upload-file")
def upload_file(audio_file: UploadFile = File(...)):
    try:
        if audio_file.filename.split(".")[-1].lower() not in valid_extenstions:
            return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "message": "Invalid file extension, valid extensions are .wav and .mp3",
                        "data": None
                    },
                )
        
        file_path = os.path.join(upload_directory, audio_file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(audio_file.file.read())

        predicted_accent, predicted_precent = predict_audio_accent(file_path)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "success": True,
                "message": "Audio classification successful",
                "data": {
                    "predicted_accent": predicted_accent if predicted_accent else "Unkown",
                    "accuracy": predicted_precent
                }
            }
        )
    
    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "message": "Something went wring please try again later",
                "data": None
            }
        )

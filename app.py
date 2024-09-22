from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from gtts import gTTS
from ocr_tamil.ocr import OCR

app = FastAPI()
ocr = OCR()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folder to save uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OCR function to extract text from an image
def extract_text_from_image(image_path):
    text_list = ocr.predict(image_path)
    return text_list[0] if text_list else ""

# Generate speech from text
def text_to_speech(text, output_file):
    if text:
        tts = gTTS(text)
        tts.save(output_file)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    extracted_text = extract_text_from_image(file_location)

    # Generate speech output
    audio_output = os.path.join(UPLOAD_FOLDER, "output_speech.mp3")
    text_to_speech(extracted_text, audio_output)

    return JSONResponse(content={
        "filename": file.filename,
        "message": "File uploaded and processed successfully",
        "extracted_text": extracted_text,
        "audio_file": f"http://127.0.0.1:8000/download-audio/"
    })

@app.get("/download-audio/")
async def download_audio():
    audio_file_path = os.path.join(UPLOAD_FOLDER, "output_speech.mp3")
    if os.path.exists(audio_file_path):
        return FileResponse(audio_file_path, media_type="audio/mpeg", filename="output_speech.mp3")
    return JSONResponse(content={"error": "Audio file not found."}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

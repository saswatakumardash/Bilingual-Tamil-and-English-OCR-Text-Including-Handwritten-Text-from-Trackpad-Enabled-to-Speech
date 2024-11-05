from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from gtts import gTTS
import requests  # For getting the ngrok URL
from tensorflow.keras.models import load_model
import uuid
app = FastAPI()
ocr = OCR(detect=True)  # Enable paragraph detection
ocr_model = load_model("ocr_tamil.h5")
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

# Function to get ngrok's public URL
def get_ngrok_url():
    try:
        response = requests.get('http://127.0.0.1:4040/api/tunnels')
        data = response.json()
        public_url = data['tunnels'][0]['public_url']
        return public_url
    except Exception as e:
        print(f"Error fetching ngrok URL: {e}")
        return None

# OCR function to extract text, including paragraphs, from an image
def extract_text_from_image(image_path):
    text_list = ocr.predict(image_path)
    paragraphs = [" ".join(text) for text in text_list]  # Combine text into paragraphs
    return "\n\n".join(paragraphs) if paragraphs else ""

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
    os.remove(file_location)  # Delete uploaded file to free memory


    # Get the ngrok public URL for sharing the audio
    public_url = get_ngrok_url()
    audio_file_url = f"https://e420-2401-4900-74e7-2fc-54b2-3f6e-8d9e-8027.ngrok-free.app/download-audio/" if public_url else "ngrok URL not available"

    return JSONResponse(content={
        "filename": file.filename,
        "message": "File uploaded and processed successfully",
        "extracted_text": extracted_text,
        "audio_file": audio_file_url
    })

@app.get("/download-audio/")
async def download_audio():
    audio_file_path = os.path.join(UPLOAD_FOLDER, "output_speech.mp3")
    if os.path.exists(audio_file_path):
        response = FileResponse(audio_file_path, media_type="audio/mpeg", filename="output_speech.mp3")
        return response
    return JSONResponse(content={"error": "Audio file not found."}, status_code=404)
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


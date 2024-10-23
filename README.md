# Bilingual (Tamil and English) OCR Text (Including Handwritten Text from Trackpad Enabled) to Audio Speech Recognition

This project enables users to convert handwritten Tamil and English text, including text drawn on a trackpad, into audio. The application uses a **CRNN model** for OCR and generates audio output using **gTTS**. Users can input text through a trackpad or upload an image, with the option for mobile users to use their camera.

## Features:
- **Trackpad Input**: Users can write Tamil or English text on the trackpad, which is then converted to audio.
- **Upload Image**: Upload images containing handwritten Tamil/English text for recognition.
- **Text-to-Speech**: Recognized text is converted to speech using **gTTS**, with future integration of **Tacotron** planned for more natural audio.
- **CRNN Model**: A custom CRNN (Convolutional Recurrent Neural Network) model with **CNN**, **LSTM**, and **GRU** layers, achieving 92% accuracy for Tamil and 95% for English characters.
  
### Live Website Link:
[https://skd-ocr-to-audio.vercel.app](https://skd-ocr-to-audio.vercel.app)

---

## Technology Stack
- **Frontend**: HTML5, CSS, JavaScript
- **Backend**: FastAPI
- **Machine Learning Model**: CRNN
- **Text-to-Speech**: gTTS
- **Deployment**: Vercel (Frontend), Ngrok (Backend)

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/saswatakumardash/Bilingual-Tamil-and-English-OCR-Text-Including-Handwritten-Text-from-Trackpad-Enabled-to-Speech.git
cd Bilingual-Tamil-and-English-OCR-Text-Including-Handwritten-Text-from-Trackpad-Enabled-to-Speech
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Ngrok
```bash
ngrok http 8000
```
**Note**: Ngrok URL needs to be updated in the frontend code (`ngrokUrl` variable in HTML) every time you start Ngrok. The backend will be available from **11:00 AM to 11:00 PM**. If you need backend access outside these hours, please email us at **saswatdashai577@gmail.com**.

### 4. Start the FastAPI Server
```bash
uvicorn app:app --reload
```

### 5. Run the Website Locally
You can run the frontend using Vercel or by serving the HTML files locally. If you deploy the frontend to Vercel, update the Ngrok URL whenever the backend restarts.

---

## How It Works

1. **Trackpad Input**: Write handwritten Tamil/English text on the trackpad interface.
2. **Upload an Image**: Upload a file containing handwritten text for recognition.
3. **OCR Model**: The CRNN model recognizes the handwritten text using CNN layers for feature extraction and LSTM/GRU layers for sequence modeling.
4. **Text-to-Speech**: The recognized text is converted to an audio file using **gTTS** and can be downloaded.

---

## Running the Website

- The live website is available at [https://skd-ocr-to-audio.vercel.app](https://skd-ocr-to-audio.vercel.app).
- **Operating Hours**: The backend is active from **11:00 AM to 11:00 PM** daily. You can access the website and use the OCR and audio features during this time.
- **Need Access Outside Operating Hours?** Email us at **saswatdashai577@gmail.com** for special arrangements.

---

## Dataset

The CRNN model was trained using a Tamil handwritten dataset from Kaggle and English Dataset([English Dataset Link](https://drive.google.com/drive/folders/1lhT6S6Y54nqG047N_MXlOacB50pb41Be?usp=drive_link). You can access the dataset from the following link:

- [Kaggle Tamil Handwritten Character Dataset](https://www.kaggle.com/datasets/gauravduttakiit/tamil-handwritten-character-recognition?select=train)

---

## Accuracy and Performance

The CRNN model was trained on a bilingual dataset (Tamil and English):
- **Tamil Accuracy**: 92%
- **English Accuracy**: 95%
- The model leverages **CNN** layers for extracting features, followed by **LSTM** and **GRU** layers for sequence processing.

### Future Enhancements:
- **Tacotron Integration**: To generate more natural-sounding speech.
- **Additional Language Support**: Expansion to other Indian languages.

---

## Project Structure

```plaintext
├── app.py                   # Backend FastAPI server
├── crnnmodel.py            # CRNN Model definition and training
├── index.html              # Frontend HTML page
├── uploads/                # Folder for uploaded files and audio output
```

---

## Screenshots

### Trackpad Input:

<img width="1440" alt="Screenshot 2024-10-23 at 23 54 38" src="https://github.com/user-attachments/assets/cfefedbc-5ad5-4203-9ecd-9a6aa8512b74">

### Audio Output:
Find the audio output in the following link:
[Audio Output Link](https://drive.google.com/file/d/1Pd7WGhPdK8Kudf-XMa9L4OAxVUQAG3VC/view?usp=sharing)
### File Upload:
![WhatsApp Image 2024-08-20 at 23 33 57](https://github.com/user-attachments/assets/5b5acaf7-147c-45ae-8fd5-211df915768b)
### File Output:
Find the audio output in the following link:
[Audio Output Link](https://drive.google.com/file/d/1d9drceGnL1ZjUtYqD9RDGvcZRKc6hDi3/view?usp=sharing)

<img width="1121" alt="Screenshot 2024-10-24 at 00 04 39" src="https://github.com/user-attachments/assets/9d5bf78d-6f90-4d3a-af5d-84751daabda3">

### Trackpad Tamil Input:
![WhatsApp Image 2024-10-24 at 00 09 43](https://github.com/user-attachments/assets/74ef9600-addd-4614-84b2-6730828c823b)

### Output: 
Find the audio output in the following link:
[Audio Output Link](https://drive.google.com/file/d/1Hu-w-XSNCGZs0qbRTlj7Vv1WYKVPFmth/view?usp=sharing)

### Camera Upload Input:
![WhatsApp Image 2024-10-24 at 00 17 20](https://github.com/user-attachments/assets/9c27ce00-37a4-4619-9782-89c0b730f2d4)

### Output:
Find the audio output in the following link:
[Audio Output Link](https://drive.google.com/file/d/1aCUONdRddMOt9KviWcJ8vAgpIZI7f0r-/view?usp=sharing)

Other inputs:
![WhatsApp Image 2024-09-22 at 22 41 12 (1)](https://github.com/user-attachments/assets/2a4d31e4-a3e4-42c7-bbd8-1f26e6804b76)


### Model Layers:
<img width="483" alt="Screenshot 2024-10-08 at 13 14 20" src="https://github.com/user-attachments/assets/94d1c17a-14ad-4773-9791-103d52e9ea99">



### CRNN Model Accuracy:
<img width="799" alt="Screenshot 2024-10-08 at 13 03 27" src="https://github.com/user-attachments/assets/bca86f53-e0af-40fd-991f-526f33aa4826">





---

## Usage Instructions

1. **Trackpad Input**: Draw handwritten Tamil or English text.
2. **Upload a File**: Upload images containing handwritten text.
3. **Audio Output**: The recognized text will be converted into an audio file available for download.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

This README includes all requested updates. Let me know if anything else is needed!

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import os
import sqlite3
from datetime import datetime
from google import genai 
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 1. INISIALISASI ---
# Dia akan langsung cari file xml yang ada di sebelah app.py
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

model_path = os.path.join("models", "interview_ai_model_v2.keras")
emotion_model = tf.keras.models.load_model(model_path)
emotions_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

def init_db():
    conn = sqlite3.connect('interview_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (timestamp TEXT, emotion TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Deteksi Wajah dengan Padding
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        # Tambah margin 10% agar dahi/dagu tidak terpotong (membantu deteksi Happy/Surprise)
        offset = int(w * 0.1)
        y1, y2 = max(0, y-offset), min(gray.shape[0], y+h+offset)
        x1, x2 = max(0, x-offset), min(gray.shape[1], x+w+offset)
        face_roi = gray[y1:y2, x1:x2]
    else:
        face_roi = gray

    # 2. Preprocessing: CLAHE (Lebih bagus dari equalizeHist untuk bayangan)
    face_roi = clahe.apply(cv2.resize(face_roi, (48, 48)))

    # 3. Normalisasi
    img_final = face_roi.astype('float32') / 255.0
    img_final = np.expand_dims(np.expand_dims(img_final, axis=-1), axis=0)
    
    # 4. Prediksi & Logic Thresholding
    prediction = emotion_model.predict(img_final)[0]
    
    # --- LOGIC PENYARING SAD vs HAPPY ---
    # Jika Happy > 25%, kita prioritaskan Happy karena Sad seringkali adalah "false positive"
    happy_index = emotions_list.index('Happy')
    sad_index = emotions_list.index('Sad')
    
    if prediction[happy_index] > 0.25:
        predicted_emotion = "Happy"
    else:
        predicted_emotion = emotions_list[np.argmax(prediction)]
    
    confidence_value = float(np.max(prediction))

    # --- 5. Suggestion (Gemini/Fallback) ---
    ai_suggestion = ""
    try:
        prompt = f"Kandidat merasa {predicted_emotion}. Berikan 1 kalimat motivasi singkat."
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        ai_suggestion = response.text.strip()
    except:
        fallbacks = {
            "Happy": "Bagus! Pertahankan senyum dan energi positifmu!", 
            "Sad": "Ayo lebih semangat! Tunjukkan antusiasmemu sedikit lagi.",
            "Neutral": "Tetap tenang dan fokus, jawabanmu terdengar profesional.",
            "Fear": "Tarik napas dalam, kamu sudah mempersiapkan ini dengan baik.",
            "Angry": "Coba lebih rileks, tunjukkan sisi ramahmu kepada interviewer.",
            "Disgust": "Alihkan fokus ke hal positif, kamu punya kendali penuh atas dirimu.",
            "Surprise": "Jangan biarkan kejutan mengganggu fokusmu, tetap tenang!"
        }
        ai_suggestion = fallbacks.get(predicted_emotion, "Tetap semangat!")

    # --- 6. Simpan DB ---
    conn = sqlite3.connect('interview_history.db')
    c = conn.cursor()
    c.execute("INSERT INTO history VALUES (?, ?, ?)", 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), predicted_emotion, confidence_value))
    conn.commit()
    conn.close()

    return {
        "emotion": predicted_emotion,
        "confidence": confidence_value,
        "motivation_quote": ai_suggestion 
    }

    # Tambahkan ini di baris paling bawah agar cPanel bisa membacanya
application = app
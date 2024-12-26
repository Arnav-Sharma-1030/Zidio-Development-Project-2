import cv2 
import sqlite3 
import smtplib 
import librosa 
import pandas as pd 
import matplotlib.pyplot as plt 
from email.mime.text import MIMEText 
from fer import FER 
from transformers import pipeline 
from cryptography.fernet import Fernet 
from sklearn.neighbors import NearestNeighbors 
# Email alert function 
def send_alert(employee_id, mood): 
    msg = MIMEText(f'Employee {employee_id} is under high stress. Current mood: {mood}') 
    msg['Subject'] = 'Stress Alert' 
    msg['From'] = 'noreply@company.com' 
    msg['To'] = 'hr@company.com' 
    with smtplib.SMTP('smtp.gmail.com', 587) as server: 
        server.starttls() 
        server.login('your_email@gmail.com', 'your_password') 
        server.sendmail('noreply@company.com', 'hr@company.com', msg.as_string()) 
# Setting up SQLite database 
conn = sqlite3.connect('employee_moods.db') 
cursor = conn.cursor() 
# Create table to store mood data 
cursor.execute(''' 
CREATE TABLE IF NOT EXISTS mood_tracking ( 
employee_id INTEGER, 
t
 imestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
mood TEXT, 
task_recommended TEXT 
) 
''') 
# Data privacy encryption key generation 
key = Fernet.generate_key() 
cipher = Fernet(key) 
# Function to encrypt data 
def encrypt_data(data): 
    return cipher.encrypt(data.encode()) 
# Function to decrypt data 
def decrypt_data(encrypted_data): 
    return cipher.decrypt(encrypted_data).decode() 
# Real-Time Emotion Detection using text (transformers) 
emotion_recognizer = pipeline("text-classification", 
    model="bhadresh-savani/bert-base-uncased-emotion") 
def detect_text_emotion(text): 
    result = emotion_recognizer(text) 
    return result[0]['label'] 
# Real-Time Emotion Detection using facial expressions (FER) 
def detect_face_emotion(frame): 
    detector = FER() 
    emotion, score = detector.top_emotion(frame) 
    return emotion, score 
# Real-Time Emotion Detection using speech (librosa) 
def extract_audio_features(audio_file): 
    y, sr = librosa.load(audio_file) 
    mfccs = librosa.feature.mfcc(y=y, sr=sr) 
    return mfccs.mean(axis=1)  # Extract mean MFCCs 
def detect_speech_emotion(audio_file): 
    features = extract_audio_features(audio_file) 
    return "neutral" 
# Task Recommendation based on detected mood 
data = pd.DataFrame({'mood': ['happy', 'stressed', 'neutral', 'happy', 'stressed'],
                     'task': ['creative', 'simple', 'moderate', 'brainstorm', 'routine'] 
                    }) 
def recommend_task(mood): 
    task_model = NearestNeighbors(n_neighbors=1) 
    task_model.fit(data[['mood']]) 
    recommended_task = task_model.kneighbors([[mood]], return_distance=False) 
    return data.iloc[recommended_task[0][0]]['task'] 
# Function to save mood data to database 
def save_mood_data(employee_id, mood, task): 
    cursor.execute("INSERT INTO mood_tracking (employee_id, mood, task_recommended) VALUES (?, ?,?)", (employee_id, mood, task)) 
    conn.commit() 
# Generate Team Mood Analytics 
def team_mood_analysis(): 
    cursor.execute("SELECT mood FROM mood_tracking") 
    moods = cursor.fetchall() 
    moods = [mood[0] for mood in moods] 
# Visualizing mood distribution 
plt.hist(moods, bins=3, edgecolor='black') 
plt.xlabel('Mood') 
plt.ylabel('Frequency') 
plt.title('Team Mood Analytics') 
plt.show() 
# Main Function to simulate employee mood detection and task recommendation 
def main(): 
# Simulating employee input 
    employee_id = 1 
    text_input = "I'm feeling overwhelmed today." 
    audio_file = "sample_audio.wav"  # Path to an audio file 
# Detect mood from text, face, and speech 
text_mood = detect_text_emotion(text_input) 
print(f"Detected mood from text: {text_mood}") 
# Simulating video capture for facial recognition 
cap = cv2.VideoCapture(0)  
ret, frame = cap.read() 
if ret: 
    face_mood, score = detect_face_emotion(frame) 
    print(f"Detected mood from face: {face_mood} with confidence {score}") 
# Detect mood from speech 
    speech_mood = detect_speech_emotion(audio_file) 
    print(f"Detected mood from speech: {speech_mood}") 
# Recommend task based on mood (using text-based mood as an example) 
    recommended_task = recommend_task(text_mood) 
    print(f"Recommended task for {text_mood} mood: {recommended_task}") 
# Save data to the database 
    save_mood_data(employee_id, text_mood, recommended_task) 
# If stressed, send an alert 
if text_mood == "stressed": 
    send_alert(employee_id, text_mood) 
# Display team mood analytics 
    team_mood_analysis() 
# Run the main function 
if __name__ == "__main__": 
    main()
# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace
import base64

app = Flask(__name__)
CORS(app)

# Emotion → Color map + advice + stress level
emotion_settings = {
    "angry": {"color": (0, 0, 255), "advice": "Take deep breaths and relax.", "stress_level": "High"},
    "disgust": {"color": (0, 100, 0), "advice": "Focus on something positive.", "stress_level": "Medium"},
    "fear": {"color": (128, 0, 128), "advice": "Stay calm, everything will be fine.", "stress_level": "High"},
    "happy": {"color": (0, 255, 0), "advice": "Keep smiling! Life loves you.", "stress_level": "Low"},
    "sad": {"color": (255, 0, 0), "advice": "Stay strong, better days are coming.", "stress_level": "Medium"},
    "surprise": {"color": (0, 165, 255), "advice": "Stay curious and open-minded.", "stress_level": "Low"},
    "neutral": {"color": (200, 200, 200), "advice": "Relax and enjoy the moment.", "stress_level": "Low"}
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        img_base64 = data['image']
        img_bytes = base64.b64decode(img_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = DeepFace.analyze(
            frame,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False
        )

        if isinstance(results, dict):
            results = [results]

        response_data = []

        for result in results:
            age = result['age']
            gender = result['dominant_gender']
            emotion = result['dominant_emotion'].lower()
            race = result['dominant_race']

            advice = emotion_settings.get(emotion, {"advice": "Be yourself!", "stress_level": "Unknown"})["advice"]
            stress_level = emotion_settings.get(emotion, {"stress_level": "Unknown"})["stress_level"]

            response_data.append({
                "age": age,
                "gender": gender,
                "emotion": emotion,
                "race": race,
                "advice": advice,
                "stress_level": stress_level
            })

        summary = {
            "detected_faces": len(results),
            "faces": response_data
        }

        return jsonify(summary)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

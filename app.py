"""
SIGN LANGUAGE READER - COMPLETE BACKEND
Features: Camera Mode + Text Mode + Analytics Dashboard + Voice Commands
"""

import cv2
import numpy as np
import base64
import mysql.connector
import pyttsx3
import threading
import time
import urllib.request
import os
import re
import json
import hashlib
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🤟 SIGN LANGUAGE READER - COMPLETE BACKEND")
print("=" * 60)

# ========== FLASK APP ==========
app = Flask(__name__)
app.secret_key = "sign_language_reader_secret_key_2024"
CORS(app, supports_credentials=True)

# ========== DOWNLOAD MEDIAPIPE MODEL ==========
model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("📥 Downloading MediaPipe model...")
    try:
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            model_path
        )
        print("✅ Model downloaded!")
    except:
        print("⚠️ Download failed")

# ========== IMPORT MEDIAPIPE ==========
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize detector
detector = None
if os.path.exists(model_path):
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        detector = vision.HandLandmarker.create_from_options(options)
        print("✅ MediaPipe initialized!")
    except Exception as e:
        print(f"⚠️ MediaPipe error: {e}")

# ========== LOAD JSON DATASET ==========
def load_gesture_dataset():
    json_path = os.path.join(os.path.dirname(__file__), "gesture_dataset.json")
    
    if not os.path.exists(json_path):
        print(f"⚠️ gesture_dataset.json not found")
        return {}, {}
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        gesture_map = {}
        speech_map = {}
        total = 0
        
        for category_name, category_data in data.items():
            for item in category_data:
                input_phrases = item.get("input", [])
                gestures = item.get("gestures", [])
                speech = item.get("speech", "")
                gesture_str = " ".join(gestures)
                
                for phrase in input_phrases:
                    gesture_map[phrase.lower()] = gesture_str
                    speech_map[phrase.lower()] = speech
                    total += 1
        
        print(f"✅ Loaded {total} phrases from JSON")
        return gesture_map, speech_map
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return {}, {}

GESTURE_MAP, SPEECH_MAP = load_gesture_dataset()

# ========== SPEECH FUNCTION ==========
speech_lock = threading.Lock()
last_spoken = {"text": None, "time": 0}
SPEECH_COOLDOWN = 1.5

def speak_gesture(gesture_name):
    now = time.time()
    if gesture_name == last_spoken["text"] and now - last_spoken["time"] < SPEECH_COOLDOWN:
        return
    
    last_spoken["text"] = gesture_name
    last_spoken["time"] = now
    
    def _speak():
        try:
            with speech_lock:
                engine = pyttsx3.init()
                engine.setProperty('rate', 140)
                engine.setProperty('volume', 1.0)
                print(f"🔊 Speaking: {gesture_name}")
                engine.say(gesture_name)
                engine.runAndWait()
                engine.stop()
        except Exception as e:
            print(f"Speech error: {e}")
    
    threading.Thread(target=_speak, daemon=True).start()

# ========== DATABASE ==========
def get_db_connection():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="sign_language_db"
        )
    except Exception as e:
        print(f"Database Error: {e}")
        return None

def log_activity(activity_type, gesture_name=None, details=None):
    """Log user activity for analytics"""
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO activity_log (activity_type, gesture_name, details) VALUES (%s, %s, %s)",
                (activity_type, gesture_name, details)
            )
            conn.commit()
            cursor.close()
            conn.close()
    except:
        pass

# ========== GESTURE CLASSIFIER (30+ GESTURES) ==========
def get_fingers_up(landmarks):
    fingers_up = []
    
    if landmarks[4].x < landmarks[3].x:
        fingers_up.append(1)
    else:
        fingers_up.append(0)
    
    if landmarks[8].y < landmarks[6].y:
        fingers_up.append(1)
    else:
        fingers_up.append(0)
    
    if landmarks[12].y < landmarks[10].y:
        fingers_up.append(1)
    else:
        fingers_up.append(0)
    
    if landmarks[16].y < landmarks[14].y:
        fingers_up.append(1)
    else:
        fingers_up.append(0)
    
    if landmarks[20].y < landmarks[18].y:
        fingers_up.append(1)
    else:
        fingers_up.append(0)
    
    return fingers_up

def classify_gesture(landmarks):
    fingers = get_fingers_up(landmarks)
    finger_count = sum(fingers)
    
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    wrist = landmarks[0]
    
    thumb_index_dist = abs(thumb_tip.x - index_tip.x) + abs(thumb_tip.y - index_tip.y)
    index_middle_dist = abs(index_tip.x - middle_tip.x) + abs(index_tip.y - middle_tip.y)
    
    # 30+ GESTURES
    if finger_count == 0:
        return "✊ FIST", 0.98, "✊", "Fist"
    
    if finger_count == 5:
        return "🖐️ OPEN PALM", 0.98, "🖐️", "Open palm"
    
    if fingers == [1, 0, 0, 0, 0]:
        return "👍 THUMBS UP", 0.97, "👍", "Thumbs up"
    
    if finger_count == 0 and thumb_tip.y > wrist.y + 0.05:
        return "👎 THUMBS DOWN", 0.95, "👎", "Thumbs down"
    
    if fingers == [0, 1, 1, 0, 0]:
        if index_middle_dist > 0.03:
            return "✌️ PEACE", 0.96, "✌️", "Peace"
        else:
            return "2️⃣ TWO", 0.91, "2️⃣", "Two"
    
    if fingers == [1, 1, 0, 0, 1]:
        return "🤟 I LOVE YOU", 0.96, "🤟", "I love you"
    
    if fingers == [1, 0, 0, 0, 1]:
        return "🤙 CALL ME", 0.94, "🤙", "Call me"
    
    if fingers == [0, 1, 0, 0, 0]:
        return "☝️ POINTING", 0.93, "☝️", "Pointing"
    
    if thumb_index_dist < 0.04 and fingers[1] == 1:
        return "👌 OK", 0.97, "👌", "Okay"
    
    if fingers == [0, 1, 0, 0, 1]:
        return "🤘 ROCK ON", 0.95, "🤘", "Rock on"
    
    if fingers == [0, 1, 1, 1, 0]:
        return "🕷️ SPIDERMAN", 0.93, "🕷️", "Spiderman"
    
    if fingers == [0, 1, 1, 1, 1]:
        return "🆘 HELP", 0.94, "🆘", "Help"
    
    if fingers == [0, 1, 1, 1, 0]:
        return "3️⃣ THREE", 0.93, "3️⃣", "Three"
    
    if fingers == [0, 1, 1, 1, 1]:
        return "4️⃣ FOUR", 0.92, "4️⃣", "Four"
    
    if fingers == [0, 0, 1, 0, 0]:
        return "🖕 MIDDLE FINGER", 0.96, "🖕", "Middle finger"
    
    if fingers == [0, 0, 0, 0, 1]:
        return "🤙 PINKY PROMISE", 0.91, "🤙", "Pinky promise"
    
    if fingers == [0, 1, 1, 0, 1]:
        return "🖖 VULCAN", 0.92, "🖖", "Live long and prosper"
    
    if fingers == [0, 1, 1, 0, 0] and index_tip.x > middle_tip.x:
        return "🤞 CROSSED", 0.90, "🤞", "Crossed fingers"
    
    if abs(index_tip.x - thumb_tip.x) < 0.02 and fingers[0] == 1 and fingers[1] == 1:
        return "❤️ HEART", 0.94, "❤️", "Heart"
    
    if fingers == [0, 1, 1, 1, 1]:
        return "🍀 GOOD LUCK", 0.89, "🍀", "Good luck"
    
    if fingers == [1, 1, 1, 0, 0] and thumb_index_dist < 0.04:
        return "💰 MONEY", 0.88, "💰", "Money"
    
    if fingers == [0, 1, 1, 0, 1]:
        return "📞 PHONE", 0.87, "📞", "Phone call"
    
    if fingers == [1, 1, 1, 1, 1] and index_tip.x < 0.3:
        return "🙏 PRAYER", 0.92, "🙏", "Prayer"
    
    if finger_count == 5 and index_tip.y < middle_tip.y:
        return "✋ STOP", 0.93, "✋", "Stop"
    
    count_names = {0: "FIST", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "OPEN PALM"}
    count_emojis = {0: "✊", 1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "🖐️"}
    count_phrases = {0: "Fist", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Open palm"}
    
    return f"{count_emojis.get(finger_count, '❓')} {count_names.get(finger_count, 'UNKNOWN')}", 0.85, count_emojis.get(finger_count, "❓"), count_phrases.get(finger_count, "Unknown")

print("✅ Gesture classifier ready (30+ gestures)")

# ========== TEXT TO GESTURE ==========
def text_to_gesture_full(text):
    text_lower = text.lower().strip()
    
    sorted_phrases = sorted(GESTURE_MAP.keys(), key=len, reverse=True)
    for phrase in sorted_phrases:
        if phrase in text_lower:
            return GESTURE_MAP[phrase], SPEECH_MAP.get(phrase, text)
    
    words = re.findall(r'\b\w+\b', text_lower)
    gestures = []
    
    for word in words:
        matched = False
        for key, value in GESTURE_MAP.items():
            if word == key or key in word or word in key:
                gestures.append(value)
                matched = True
                break
        if not matched:
            gestures.append(f"[{word.upper()}]")
    
    if gestures:
        return " → ".join(gestures), text
    
    return f"❓ {text.upper()}", text

# ========== COOLDOWN SYSTEM ==========
last_detected = {"gesture": None, "time": 0}
COOLDOWN_DETECT = 2.5

def can_detect(gesture):
    now = time.time()
    if gesture != last_detected["gesture"]:
        last_detected["gesture"] = gesture
        last_detected["time"] = now
        return True
    if now - last_detected["time"] >= COOLDOWN_DETECT:
        last_detected["time"] = now
        return True
    return False

# ========== API ENDPOINTS ==========

# ===== EXISTING ENDPOINTS =====
@app.route('/')
def home():
    return jsonify({"status": "running", "phrases_loaded": len(GESTURE_MAP), "gestures": 30})

@app.route('/text-to-gesture', methods=['POST'])
def text_to_gesture_api():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        gesture, speech = text_to_gesture_full(text)
        
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO gesture_history (gesture_name, confidence, mode, input_text) VALUES (%s, %s, %s, %s)",
                (gesture, 99.0, 'text', text)
            )
            conn.commit()
            cursor.close()
            conn.close()
        
        log_activity('text_conversion', None, f'Converted: {text}')
        speak_gesture(speech)
        
        emoji = "🤟"
        emoji_list = re.findall(r'[\U0001F300-\U0001F9FF]', gesture)
        if emoji_list:
            emoji = emoji_list[0]
        
        return jsonify({
            "success": True,
            "input_text": text,
            "gesture": gesture,
            "emoji": emoji,
            "speech_text": speech,
            "message": f"✅ Converted: {text}"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data['image']
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        if not detector:
            return jsonify({"success": False, "error": "Detector not initialized"}), 500
        
        result = detector.detect(mp_image)
        
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            hand_landmarks = result.hand_landmarks[0]
            gesture, confidence, emoji, speech = classify_gesture(hand_landmarks)
            
            if can_detect(gesture):
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO gesture_history (gesture_name, confidence, mode) VALUES (%s, %s, %s)",
                        (gesture, confidence * 100, 'camera')
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()
                    print(f"✅ Saved to DB: {gesture}")
                
                log_activity('camera_detection', gesture, f'Detected with {confidence*100:.1f}% confidence')
                speak_gesture(speech)
                
                return jsonify({
                    "success": True,
                    "gesture": gesture,
                    "display_text": gesture,
                    "emoji": emoji,
                    "confidence": round(confidence * 100, 2),
                    "message": f"Recognized: {gesture}",
                    "new_detection": True
                })
            else:
                return jsonify({
                    "success": True,
                    "gesture": gesture,
                    "display_text": gesture,
                    "emoji": emoji,
                    "confidence": round(confidence * 100, 2),
                    "message": f"Same gesture - cooldown",
                    "new_detection": False
                })
        else:
            return jsonify({
                "success": True,
                "gesture": "❌ NO HAND",
                "display_text": "No Hand Detected",
                "emoji": "❌",
                "confidence": 0,
                "message": "Show your hand to the camera",
                "new_detection": False
            })
            
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"success": True, "history": []})
        
        cursor = conn.cursor()
        cursor.execute("SELECT gesture_name, confidence, mode, input_text, timestamp FROM gesture_history ORDER BY timestamp DESC LIMIT 50")
        rows = cursor.fetchall()
        history = []
        for row in rows:
            history.append({
                "gesture_name": row[0],
                "confidence": row[1],
                "mode": row[2],
                "input_text": row[3],
                "timestamp": str(row[4])
            })
        cursor.close()
        conn.close()
        
        return jsonify({"success": True, "history": history})
    except Exception as e:
        return jsonify({"success": True, "history": []})

@app.route('/clear_history', methods=['DELETE'])
def clear_history():
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM gesture_history")
            conn.commit()
            cursor.close()
            conn.close()
        return jsonify({"success": True, "message": "History cleared"})
    except:
        return jsonify({"success": True, "message": "History cleared"})

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"success": True, "total": 0, "camera": 0, "text": 0})
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM gesture_history")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gesture_history WHERE mode = 'camera'")
        camera = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gesture_history WHERE mode = 'text'")
        text = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return jsonify({"success": True, "total": total, "camera": camera, "text": text})
    except:
        return jsonify({"success": True, "total": 0, "camera": 0, "text": 0})

# ===== ANALYTICS DASHBOARD ENDPOINTS =====
@app.route('/analytics/overview', methods=['GET'])
def get_analytics_overview():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"success": False, "error": "Database error"})
        
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM gesture_history")
        total_gestures = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gesture_history WHERE DATE(timestamp) = CURDATE()")
        today_gestures = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT gesture_name, COUNT(*) as count 
            FROM gesture_history 
            GROUP BY gesture_name 
            ORDER BY count DESC 
            LIMIT 1
        """)
        most_used = cursor.fetchone()
        most_used_gesture = most_used[0] if most_used else "None"
        most_used_count = most_used[1] if most_used else 0
        
        cursor.execute("SELECT COUNT(*) FROM gesture_history WHERE mode = 'camera'")
        camera_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM gesture_history WHERE mode = 'text'")
        text_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confidence) FROM gesture_history WHERE confidence > 0")
        avg_confidence = cursor.fetchone()[0] or 0
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "success": True,
            "total_gestures": total_gestures,
            "today_gestures": today_gestures,
            "most_used_gesture": most_used_gesture,
            "most_used_count": most_used_count,
            "camera_count": camera_count,
            "text_count": text_count,
            "avg_confidence": round(avg_confidence, 2)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/analytics/chart-data', methods=['GET'])
def get_chart_data():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"success": False, "error": "Database error"})
        
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count 
            FROM gesture_history 
            WHERE timestamp >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
            GROUP BY DATE(timestamp)
            ORDER BY date ASC
        """)
        daily_data = cursor.fetchall()
        
        dates = [str(row[0]) for row in daily_data]
        counts = [row[1] for row in daily_data]
        
        cursor.execute("""
            SELECT gesture_name, COUNT(*) as count 
            FROM gesture_history 
            GROUP BY gesture_name 
            ORDER BY count DESC 
            LIMIT 5
        """)
        top_gestures = cursor.fetchall()
        
        top_gesture_names = [row[0][:20] for row in top_gestures]
        top_gesture_counts = [row[1] for row in top_gestures]
        
        cursor.execute("""
            SELECT HOUR(timestamp) as hour, COUNT(*) as count 
            FROM gesture_history 
            GROUP BY HOUR(timestamp)
            ORDER BY hour ASC
        """)
        hourly_data = cursor.fetchall()
        
        hours = [row[0] for row in hourly_data]
        hourly_counts = [row[1] for row in hourly_data]
        
        cursor.close()
        conn.close()
        
        return jsonify({
            "success": True,
            "dates": dates,
            "daily_counts": counts,
            "top_gesture_names": top_gesture_names,
            "top_gesture_counts": top_gesture_counts,
            "hours": hours,
            "hourly_counts": hourly_counts
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/analytics/activity-log', methods=['GET'])
def get_activity_log():
    try:
        limit = request.args.get('limit', 50)
        conn = get_db_connection()
        if not conn:
            return jsonify({"success": False, "error": "Database error"})
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT activity_type, gesture_name, details, timestamp 
            FROM activity_log 
            ORDER BY timestamp DESC 
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        
        activities = []
        for row in rows:
            activities.append({
                "type": row[0],
                "gesture": row[1],
                "details": row[2],
                "time": str(row[3])
            })
        
        cursor.close()
        conn.close()
        
        return jsonify({"success": True, "activities": activities})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ===== VOICE COMMANDS ENDPOINTS =====
@app.route('/voice/execute', methods=['POST'])
def execute_voice_command():
    try:
        data = request.json
        command = data.get('command', '').lower().strip()
        
        print(f"🎤 Voice command received: {command}")
        
        if 'camera' in command or 'start camera' in command:
            return jsonify({"success": True, "action": "navigate", "url": "camera_mode.html", "message": "Opening Camera Mode"})
        
        elif 'text' in command or 'text mode' in command or 'type' in command:
            return jsonify({"success": True, "action": "navigate", "url": "text_mode.html", "message": "Opening Text/Speech Mode"})
        
        elif 'dashboard' in command or 'analytics' in command or 'stats' in command:
            return jsonify({"success": True, "action": "navigate", "url": "dashboard.html", "message": "Opening Analytics Dashboard"})
        
        elif 'history' in command and ('delete' in command or 'clear' in command):
            return jsonify({"success": True, "action": "clear_history", "message": "Clearing history..."})
        
        elif 'history' in command or 'show history' in command or 'view history' in command:
            return jsonify({"success": True, "action": "show_history", "message": "Fetching history..."})
        
        elif 'back' in command or 'return' in command or 'home' in command or 'welcome' in command:
            return jsonify({"success": True, "action": "navigate", "url": "welcome.html", "message": "Returning to Home"})
        
        elif 'refresh' in command or 'fresh' in command:
            return jsonify({"success": True, "action": "refresh", "message": "Refreshing data..."})
        
        elif 'stop listening' in command or 'stop now' in command or 'stop' in command:
            return jsonify({"success": True, "action": "stop_listening", "message": "Stopping voice recognition"})
        
        elif 'speak' in command or 'say it' in command or 'read' in command:
            return jsonify({"success": True, "action": "speak", "message": "Speaking current content..."})
        
        elif 'help' in command or 'what can i say' in command or 'commands' in command:
            return jsonify({"success": True, "action": "show_help", "message": "Showing help menu"})
        
        else:
            return jsonify({"success": False, "action": "unknown", "message": f"Command '{command}' not recognized. Say 'Help' for available commands."})
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/voice/history', methods=['GET'])
def get_voice_history():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"success": True, "history": []})
        
        cursor = conn.cursor()
        cursor.execute("SELECT gesture_name, confidence, mode, timestamp FROM gesture_history ORDER BY timestamp DESC LIMIT 20")
        rows = cursor.fetchall()
        history = []
        for row in rows:
            history.append({
                "gesture": row[0],
                "confidence": row[1],
                "mode": row[2],
                "time": str(row[3])
            })
        cursor.close()
        conn.close()
        
        return jsonify({"success": True, "history": history})
    except:
        return jsonify({"success": True, "history": []})

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("✅ COMPLETE SERVER READY!")
    print("=" * 60)
    print(f"📚 Loaded {len(GESTURE_MAP)} phrases from JSON")
    print("📍 API: http://localhost:5000")
    print("")
    print("🎯 FEATURES:")
    print("   📹 Camera Mode - 30+ gestures")
    print("   ⌨️ Text/Speech Mode - 350+ phrases")
    print("   📊 Analytics Dashboard - Charts & Stats")
    print("   🗣️ Voice Commands - Hands-free navigation")
    print("")
    print("🌐 OPEN: http://localhost/sign_language_web/welcome.html")
    print("=" * 60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
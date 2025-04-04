from flask import Flask, render_template, Response, request
import cv2
import os
from deepface import DeepFace

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize global camera variable
camera = None  

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion-to-message mapping
positive_messages = {
    "happy": "Keep smiling! Your happiness is contagious! 😊",
    "sad": "You are stronger than you think! Better days are coming! ❤",
    "angry": "Take a deep breath! Stay calm and shine brighter! ✨",
    "fear": "Courage conquers all fears! You’ve got this! 💪",
    "surprise": "Life is full of wonderful surprises! Keep exploring! 🎉",
    "neutral": "Stay balanced and positive! Every day is a new chance! 🌟",
    "disgust": "Focus on the good! There’s always something beautiful! 🌸"
}

# Slideshow images
slideshow_images = ["static/slideshow/child.jpg", "static/slideshow/girls.jpg", "static/slideshow/group.jpg"]

@app.route('/')
def index():
    return render_template('index.html', slideshow_images=slideshow_images)

@app.route('/upload')
def upload():
    return render_template('upload.html', slideshow_images=slideshow_images)

@app.route('/about')
def about():
    return render_template('about.html', slideshow_images=slideshow_images)

@app.route('/video_feed')
def video_feed():
    """ Stream real-time video with face and emotion detection. """
    def generate_frames():
        global camera
        if camera is None:
            camera = cv2.VideoCapture(0)  # Open camera only when needed

        while camera.isOpened():
            success, frame = camera.read()
            if not success:
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(40, 40))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]  # Extract face ROI

                # Analyze emotion
                try:
                    result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']

                    # Draw rectangle & text
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"Error analyzing emotion: {e}")

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection')
def stop_detection():
    """ Properly release the webcam when stopping detection. """
    global camera
    if camera is not None:
        camera.release()
        camera = None  # Reset camera object
    cv2.destroyAllWindows()
    return "Detection Stopped"

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    """ Detects emotions from an uploaded image. """
    if 'image' not in request.files:
        return "No file uploaded", 400

    image = request.files['image']
    
    if image.filename == '':
        return "No selected file", 400

    # Save the uploaded image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    # Load image
    img = cv2.imread(image_path)

    try:
        # Analyze emotion
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        message = positive_messages.get(emotion, "You're amazing! 🌈")
        
        return render_template('upload.html', emotion=emotion, message=message, image_filename=image.filename, slideshow_images=slideshow_images)
    
    except Exception as e:
        return f"Error analyzing emotion: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)

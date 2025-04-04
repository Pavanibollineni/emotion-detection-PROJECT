import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion-to-message mapping
positive_messages = {
    "happy": "Keep smiling! Your happiness is contagious! 😊",
    "sad": "You are stronger than you think! Better days are coming! ❤",
    "angry": "Take a deep breath! Stay calm and shine brighter!!!!!!!!!!!!!! ✨",
    "fear": "Courage conquers all fears! You’ve got this! 💪",
    "surprise": "Life is full of wonderful surprises! Keep exploring!!!!!!!!!!! 🎉",
    "neutral": "Stay balanced and positive! Every day is a new chance !!!!!!🌟",
    "disgust": "Focus on the good! There’s always something beautiful!!!!!!!!!! 🌸"
}

while True:
    ret, frame = cap.read("orya")
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(40, 40))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Extract face ROI

        # Analyze emotion
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)  # Use full frame
            emotion = result[0]['dominant_emotion']
            message = positive_messages.get(emotion, "You are amazing! Keep going! 🌈")

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Display emotion and message
            cv2.putText(frame, emotion.capitalize(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, message, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        except Exception as e:
            print(f"Error: {e}")

    # Show the frame
    cv2.imshow("Real-Time Emotion Detection with Positive Messages", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
cv2.destroyallwindows()
cap.release()

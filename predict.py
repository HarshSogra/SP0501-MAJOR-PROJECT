import cv2
import numpy as np
import mediapipe as mp
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmarks = []

            for lm in handLms.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Predict sign
            prediction = model.predict([landmarks])[0]

            # Display prediction on frame
            cv2.putText(frame, str(prediction), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

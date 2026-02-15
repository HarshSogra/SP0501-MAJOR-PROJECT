import cv2
import numpy as np
import mediapipe as mp
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

# Give image path here
image_path = "test.jpg"   # 🔹 Change this to your image file name

# Read image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Convert BGR to RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image
results = hands.process(rgb)

if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
        landmarks = []

        for lm in handLms.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

        # Predict sign
        prediction = model.predict([landmarks])[0]

        # Display prediction on image
        cv2.putText(image, prediction, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    print("Prediction:", prediction)

else:
    print("No hand detected!")

# Show output image
cv2.imshow("Sign Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

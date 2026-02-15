import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
data = []
label = "THANKS"  # Change this to "YES", "NO", "THANKS" etc. for each run

print(f"Collecting data for: {label}")
print("Show your hand clearly, then press 's' to save ONE sample")
print("Press 'q' to finish and save the data")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Show status on screen
    if results.multi_hand_landmarks:
        cv2.putText(frame, "Hand detected - Press 's' to capture", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Show your hand", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Samples collected: {len(data)}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Collect Data - Press 's' to save sample", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 's' to save current hand pose (only if hand is detected)
    if key == ord('s') and results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]  # Take the first hand
        landmarks = []
        for lm in hand.landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)
        data.append(landmarks)
        print(f"Captured sample #{len(data)} for {label}")

    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save only when quitting
if data:
    np.save(f"{label}.npy", np.array(data))
    print(f"\nDone! Saved {len(data)} samples to {label}.npy")
else:
    print("No data collected.")
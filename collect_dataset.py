import cv2
import os

print("Program started...")

# ==============================
# SETTINGS
# ==============================

DATASET_DIR = "captured_data"
CLASS_NAME = "HELLO"   # Change this for each sign
NUM_IMAGES = 200

save_path = os.path.join(DATASET_DIR, CLASS_NAME)

if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"Collecting {NUM_IMAGES} images for class: {CLASS_NAME}")

# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not detected")
    exit()

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    # Draw capture box
    x1, y1 = 300, 100
    x2, y2 = 600, 400

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    roi = frame[y1:y2, x1:x2]

    cv2.putText(frame,
                f"Images Collected: {count}/{NUM_IMAGES}",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1)

    # Press S to capture
    if key == ord('s'):
        file_name = os.path.join(save_path, f"{CLASS_NAME}_{count}.png")
        cv2.imwrite(file_name, roi)
        count += 1
        print(f"Saved image {count}")

    # Stop when enough images collected
    if count >= NUM_IMAGES:
        break

    # Press Q to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset collection completed!")
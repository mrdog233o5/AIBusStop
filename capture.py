import cv2
import os
import glob

class CONFIG:
    WIDTH = 720
    HEIGHT = 720
    DATASET_PATH = "dataset"
    IMAGE_PREFIX = "bus_"

def get_next_image_number(folder, prefix):
    pattern = os.path.join(folder, f"{prefix}*.jpg")
    files = glob.glob(pattern)
    if not files:
        return 0
    numbers = []
    for f in files:
        name = os.path.basename(f)
        try:
            numbers.append(int(name[len(prefix):-4]))
        except:
            continue
    return max(numbers) + 1 if numbers else 0

os.makedirs(CONFIG.DATASET_PATH, exist_ok=True)
img_count = get_next_image_number(CONFIG.DATASET_PATH, CONFIG.IMAGE_PREFIX)

cap = cv2.VideoCapture(1)  # change to 0 if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.HEIGHT)

print("SPACE to capture, q to quit. Next number:", img_count)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Capture', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:
        filename = f"{CONFIG.IMAGE_PREFIX}{img_count:04d}.jpg"
        path = os.path.join(CONFIG.DATASET_PATH, filename)
        cv2.imwrite(path, frame)
        print(f"Saved {filename}")
        img_count += 1
        cv2.putText(frame, "CAPTURED!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Capture', frame)
        cv2.waitKey(300)

cap.release()
cv2.destroyAllWindows()
import cv2
import os
import glob
from ultralytics import YOLO
import easyocr

class CONFIG:
    # Camera settings
    CAM_NUM = 0
    WIDTH = 720
    HEIGHT = 720
    DATASET_PATH = "dataset"

    # Detection settings (set after training)
    MODEL_PATH = "best.pt"          # Your trained model
    USE_OCR = True                   # Enable route number reading
    CONF_THRESHOLD = 0.5             # Minimum confidence for detections

class Sampler:
    def __init__(self):
        self.image_count = self.get_next_image_number("")

    def capturePhoto(self, app):
        filename = f"{self.image_count:04d}.jpg"
        filepath = os.path.join(CONFIG.DATASET_PATH, filename)
        cv2.imwrite(filepath, app.frame)
        print(f"Saved: {filename}")
        self.image_count += 1
        cv2.putText(app.frame, "CAPTURED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Bus Camera Test', app.frame)
        cv2.waitKey(300)

    def get_next_image_number(self, prefix):
        pattern = os.path.join(CONFIG.DATASET_PATH, f"{prefix}*.jpg")
        existing = glob.glob(pattern)
        if not existing:
            return 0
        numbers = []
        for fp in existing:
            fname = os.path.basename(fp)
            num_str = fname[len(prefix):-4]
            try:
                numbers.append(int(num_str))
            except ValueError:
                continue
        return max(numbers) + 1 if numbers else 0

class AIBusStop:
    def __init__(self, mode="collect"):   # "collect" or "detect"
        self.mode = mode
        self.frame = None
        os.makedirs(CONFIG.DATASET_PATH, exist_ok=True)

        # Camera
        self.cap = cv2.VideoCapture(CONFIG.CAM_NUM)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.HEIGHT)

        if mode == "collect":
            self.sampler = Sampler()
            self.model = None
            self.reader = None
        else:  # detect mode
            print("Loading YOLO model...")
            self.model = YOLO(CONFIG.MODEL_PATH)
            if CONFIG.USE_OCR:
                print("Loading EasyOCR...")
                self.reader = easyocr.Reader(['en'])
            self.sampler = None

    def update(self):
        ret, self.frame = self.cap.read()
        if not ret:
            return False

        if self.mode == "collect":
            cv2.imshow('Bus Camera Test', self.frame)
        else:
            # Run detection
            results = self.model(self.frame, conf=CONFIG.CONF_THRESHOLD, conf=0.7)[0]
            annotated = results.plot()  # draws boxes with labels

            # If OCR is enabled, crop each route_number and read it
            if CONFIG.USE_OCR and self.reader:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls == 1:  # route_number class ID
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = self.frame[y1:y2, x1:x2]
                        ocr_result = self.reader.readtext(crop)
                        if ocr_result:
                            text = ocr_result[0][1]
                            # Put text near the box
                            cv2.putText(annotated, text, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            cv2.imshow('Bus Detection', annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('m'):          # press 'm' to switch mode
            self.mode = "detect" if self.mode == "collect" else "collect"
            print(f"Switched to {self.mode} mode")
            # Re‑initialise mode‑specific components
            if self.mode == "detect" and self.model is None:
                self.model = YOLO(CONFIG.MODEL_PATH)
                if CONFIG.USE_OCR:
                    self.reader = easyocr.Reader(['en'])
            return True
        elif self.mode == "collect" and key == 32:  # spacebar
            self.sampler.capturePhoto(self)

        return True

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    # Start in collect mode (you can change to "detect" after training)
    busStop = AIBusStop(mode="detect")
    while busStop.update():
        pass
    busStop.cleanup()

if __name__ == "__main__":
    main()
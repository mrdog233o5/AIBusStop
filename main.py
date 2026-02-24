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

    # Detection settings
    MODEL_PATH = "best.pt"
    USE_OCR = True
    CONF_THRESHOLD = 0.5             # YOLO confidence threshold

    # OCR settings (new)
    OCR_CONF_THRESHOLD = 0.4          # Minimum confidence for OCR result
    PREPROCESS_OCR = True              # Enable preprocessing (resize, threshold)
    OCR_RESIZE_FACTOR = 2               # How much to enlarge the cropped image
    OCR_THRESHOLD_VALUE = 120           # Threshold value for binary conversion

class Sampler:
    # (unchanged)
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
    def __init__(self, mode="collect"):
        self.mode = mode
        self.frame = None
        os.makedirs(CONFIG.DATASET_PATH, exist_ok=True)

        self.cap = cv2.VideoCapture(CONFIG.CAM_NUM)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.HEIGHT)

        if mode == "collect":
            self.sampler = Sampler()
            self.model = None
            self.reader = None
        else:
            print("Loading YOLO model...")
            self.model = YOLO(CONFIG.MODEL_PATH)
            if CONFIG.USE_OCR:
                print("Loading EasyOCR...")
                self.reader = easyocr.Reader(['en'])
            self.sampler = None

    def preprocess_for_ocr(self, crop):
        """Convert to grayscale, resize, and apply threshold."""
        if crop.size == 0:
            return None
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Resize to make text larger
        h, w = gray.shape
        gray = cv2.resize(gray, (w * CONFIG.OCR_RESIZE_FACTOR, h * CONFIG.OCR_RESIZE_FACTOR),
                          interpolation=cv2.INTER_CUBIC)
        # Apply binary threshold
        _, thresh = cv2.threshold(gray, CONFIG.OCR_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        return thresh

    def update(self):
        ret, self.frame = self.cap.read()
        if not ret:
            return False

        if self.mode == "collect":
            cv2.imshow('Bus Camera Test', self.frame)
        else:
            # Run detection
            results = self.model(self.frame, conf=CONFIG.CONF_THRESHOLD)[0]
            annotated = results.plot()

            # OCR on each route_number detection
            if CONFIG.USE_OCR and self.reader:
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls == 1:  # route_number class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = self.frame[y1:y2, x1:x2]

                        # Preprocess the cropped image
                        if CONFIG.PREPROCESS_OCR:
                            processed_crop = self.preprocess_for_ocr(crop)
                            if processed_crop is None:
                                continue
                        else:
                            processed_crop = crop

                        # Run OCR
                        ocr_result = self.reader.readtext(processed_crop)

                        # Filter by confidence and display
                        if ocr_result and ocr_result[0][2] > CONFIG.OCR_CONF_THRESHOLD:
                            text = ocr_result[0][1]
                            confidence = ocr_result[0][2]
                            display_text = f"{text} ({confidence:.2f})"
                            color = (0, 255, 0)  # green for confident
                        else:
                            display_text = "?"
                            color = (0, 0, 255)  # red for unsure

                        cv2.putText(annotated, display_text, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Bus Detection', annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('m'):
            self.mode = "detect" if self.mode == "collect" else "collect"
            print(f"Switched to {self.mode} mode")
            if self.mode == "detect" and self.model is None:
                self.model = YOLO(CONFIG.MODEL_PATH)
                if CONFIG.USE_OCR:
                    self.reader = easyocr.Reader(['en'])
            return True
        elif self.mode == "collect" and key == 32:
            self.sampler.capturePhoto(self)

        return True

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    busStop = AIBusStop(mode="detect")
    while busStop.update():
        pass
    busStop.cleanup()

if __name__ == "__main__":
    main()
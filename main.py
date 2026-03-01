import cv2
import os
import glob
import time
from ultralytics import YOLO
import easyocr

class CONFIG:
    CAM_NUM = 0
    WIDTH = 720
    HEIGHT = 720
    DATASET_PATH = "dataset"

    MODEL_PATH = "best.pt"
    USE_OCR = True
    CONF_THRESHOLD = 0.5

    OCR_CONF_THRESHOLD = 0.3
    PREPROCESS_OCR = True
    OCR_RESIZE_FACTOR = 2
    OCR_THRESHOLD_VALUE = 60

    SAVE_CROP_IMAGES = False          # Set to True only for debugging

class Sampler:
    def __init__(self):
        self.image_count = self.get_next_image_number("")

    def capturePhoto(self, app):
        filename = f"{self.image_count:04d}.jpg"
        filepath = os.path.join(CONFIG.DATASET_PATH, filename)
        cv2.imwrite(filepath, app.frame)
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

        self.bus_log = {}
        self.bus_last_seen = {}
        self.seen_timeout = 5.0

        if mode == "collect":
            self.sampler = Sampler()
            self.model = None
            self.reader = None
        else:
            self.model = YOLO(CONFIG.MODEL_PATH)
            if CONFIG.USE_OCR:
                self.reader = easyocr.Reader(['en'], verbose=False)
            self.sampler = None

    def normalise_ocr_text(self, raw_text):
        """Clean OCR output: keep only letters/digits, uppercase, and fix common errors."""
        clean = ''.join(c for c in raw_text if c.isalnum()).upper()
        corrections = {
            "S": "5",
            "O": "0",
        }
        return corrections.get(clean, clean)

    def preprocess_for_ocr(self, crop, timestamp, idx):
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if CONFIG.SAVE_CROP_IMAGES:
            cv2.imwrite(f"debug_gray_{timestamp}_{idx}.jpg", gray)
        h, w = gray.shape
        gray = cv2.resize(gray, (w * CONFIG.OCR_RESIZE_FACTOR, h * CONFIG.OCR_RESIZE_FACTOR),
                          interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, CONFIG.OCR_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        if CONFIG.SAVE_CROP_IMAGES:
            cv2.imwrite(f"debug_prep_{timestamp}_{idx}.jpg", thresh)
        return thresh

    def update(self):
        ret, self.frame = self.cap.read()
        if not ret:
            return False

        if self.mode == "collect":
            cv2.imshow('Bus Camera Test', self.frame)
        else:
            # Run detection with verbose=False to suppress YOLO logs
            results = self.model(self.frame, conf=CONFIG.CONF_THRESHOLD, verbose=False)[0]
            annotated = results.plot()

            current_numbers = []   # (x_center, text)

            if CONFIG.USE_OCR and self.reader:
                for idx, box in enumerate(results.boxes):
                    cls = int(box.cls[0])
                    if cls == 1:  # route_number class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x_center = (x1 + x2) // 2
                        crop = self.frame[y1:y2, x1:x2]
                        timestamp = int(time.time())

                        if CONFIG.SAVE_CROP_IMAGES:
                            cv2.imwrite(f"debug_crop_{timestamp}_{idx}.jpg", crop)

                        if CONFIG.PREPROCESS_OCR:
                            processed_crop = self.preprocess_for_ocr(crop, timestamp, idx)
                            if processed_crop is None:
                                continue
                        else:
                            processed_crop = crop

                        ocr_result = self.reader.readtext(processed_crop)

                        # Default display values
                        display_text = "?"
                        color = (0, 0, 255)

                        if ocr_result and ocr_result[0][2] > CONFIG.OCR_CONF_THRESHOLD:
                            raw_text = ocr_result[0][1]
                            text = self.normalise_ocr_text(raw_text)
                            if text:  # only use if cleaning produced something
                                display_text = f"{text} ({ocr_result[0][2]:.2f})"
                                color = (0, 255, 0)
                                current_numbers.append((x_center, text))

                                current_time = time.time()
                                if text not in self.bus_log:
                                    self.bus_log[text] = current_time
                                self.bus_last_seen[text] = current_time

                        cv2.putText(annotated, display_text, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Print left‑to‑right order for this frame
            if current_numbers:
                current_numbers.sort(key=lambda x: x[0])
                order_line = " ".join(text for _, text in current_numbers)
                print(order_line)

            # Remove old buses (silent)
            now = time.time()
            to_remove = [route for route, last in self.bus_last_seen.items() if now - last > self.seen_timeout]
            for route in to_remove:
                del self.bus_log[route]
                del self.bus_last_seen[route]

            cv2.imshow('Bus Detection', annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('m'):
            self.mode = "detect" if self.mode == "collect" else "collect"
            if self.mode == "detect" and self.model is None:
                self.model = YOLO(CONFIG.MODEL_PATH)
                if CONFIG.USE_OCR:
                    self.reader = easyocr.Reader(['en'], verbose=False)
            return True
        elif key == ord('o'):
            if self.mode == "detect":
                if not self.bus_log:
                    print("No buses have been detected yet.")
                else:
                    sorted_buses = sorted(self.bus_log.items(), key=lambda x: x[1])
                    print("\n--- Bus Arrival Order ---")
                    for idx, (route, first_time) in enumerate(sorted_buses, 1):
                        print(f"{idx}. Route {route} at {time.ctime(first_time)}")
                    print("------------------------")
            else:
                print("Order only available in detect mode")
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


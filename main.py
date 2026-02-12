import cv2
import os
import glob

class CONFIG:
    CAM_NUM = 0
    WIDTH = 720
    HEIGHT = 720
    DATASET_PATH = "dataset"

class Sampler:
    def __init__(self):
        self.image_count = 0
        self.image_count = self.get_next_image_number("")

    def capturePhoto(self, app):
        # Save the image
        filename = f"{self.image_count:04d}.jpg"
        filepath = os.path.join(CONFIG.DATASET_PATH, filename)
        cv2.imwrite(filepath, app.frame)
        print(f"Saved: {filename}")
        self.image_count += 1
        
        # Show a quick confirmation on screen
        cv2.putText(app.frame, "CAPTURED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Bus Camera Test', app.frame)
        cv2.waitKey(300)

    def get_next_image_number(self, prefix):
        pattern = os.path.join(CONFIG.DATASET_PATH, f"{prefix}*.jpg")
        
        existing_files = glob.glob(pattern)
        
        if not existing_files:
            return 0
        
        numbers = []
        for filepath in existing_files:
            filename = os.path.basename(filepath)
            number_str = filename[len(prefix):-4]
            try:
                numbers.append(int(number_str))
            except ValueError:
                continue
        
        if not numbers:
            return 0
        
        return max(numbers) + 1


class AIBusStop:
    def __init__(self):
        self.cap = None 

        # create dataset folder
        os.makedirs(CONFIG.DATASET_PATH, exist_ok=True)

        # create cv2 capture window
        self.cap = cv2.VideoCapture(CONFIG.CAM_NUM)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.HEIGHT)

        self.sampler = Sampler()
    
    def update(self):
        ret, self.frame = self.cap.read()
        
        if ret:
            cv2.imshow('Bus Camera Test', self.frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == 32:
            self.sampler.capturePhoto(self)

        return True
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    busStop = AIBusStop()

    cont = True
    while cont:
        cont = busStop.update()

    busStop.cleanup()

if __name__ == "__main__":
    main()
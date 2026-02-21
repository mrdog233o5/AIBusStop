import cv2
from ultralytics import YOLO

# Load the pre-trained model (nano version – fast, small)
model = YOLO("yolov8n.pt")  # Downloads automatically the first time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)  # verbose=False hides extra prints
    
    # Draw results on the frame
    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLO Bus Detector', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
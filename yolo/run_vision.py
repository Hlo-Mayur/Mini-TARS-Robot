import cv2
import argparse
from ultralytics import YOLO

def get_decision_from_zones(detected_zones):
    """Makes a navigation decision based on which zones are occupied."""
    command = ""
    if "CENTER" in detected_zones:
        command = "Decision: STOP"
    elif "LEFT" in detected_zones and "RIGHT" in detected_zones:
        command = "Decision: STOP - Obstacles on both sides"
    elif "LEFT" in detected_zones:
        command = "Decision: TURN RIGHT"
    elif "RIGHT" in detected_zones:
        command = "Decision: TURN LEFT"
    else:
        command = "Decision: CLEAR TO MOVE"
    return command

# --- Argument Parsing ---
# This makes the script more flexible by allowing you to specify the model from the command line
parser = argparse.ArgumentParser(description="Run YOLO object detection on a webcam feed.")
parser.add_argument("--model", type=str, default="best1.pt", help="Path to the YOLO model file.")
args = parser.parse_args()
model_path = args.model

# --- Model Loading ---
try:
    model = YOLO(model_path)
    print(f"Successfully loaded model: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Webcam Initialization ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# --- Main Loop ---
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    # Get frame dimensions
    H, W, _ = frame.shape
    
    # Define the zone boundaries
    zone_1_boundary = W // 3
    zone_2_boundary = 2 * W // 3

    # Run YOLO inference with a confidence threshold
    results = model(frame, conf=0.5)

    # Get the annotated frame with bounding boxes
    annotated_frame = results[0].plot()

    # Draw zone lines for visualization
    cv2.line(annotated_frame, (zone_1_boundary, 0), (zone_1_boundary, H), (255, 0, 0), 2) # Blue line
    cv2.line(annotated_frame, (zone_2_boundary, 0), (zone_2_boundary, H), (0, 0, 255), 2) # Red line

    # --- Decision Logic ---
    detected_zones = set()
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            box_center_x = (x1 + x2) / 2
            
            if box_center_x < zone_1_boundary:
                detected_zones.add("LEFT")
            elif box_center_x < zone_2_boundary:
                detected_zones.add("CENTER")
            else:
                detected_zones.add("RIGHT")
    
    # Get the final command from our function
    command = get_decision_from_zones(detected_zones)
    
    # Print the command to the terminal
    print(command)

    # Display the frame
    cv2.imshow("TARS Vision System", annotated_frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
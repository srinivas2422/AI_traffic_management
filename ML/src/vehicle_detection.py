from ultralytics import YOLO  # Import the YOLO class
def detect_vehicles(frame_path, model_path="../models/yolo11n.pt"):
    """Detect vehicles in an image using YOLO and return vehicle count and bounding boxes."""
    # Load the YOLO model
    yolo_model = YOLO(model_path)
    
    # Perform detection
    results = yolo_model(frame_path)
    vehicle_boxes = results[0].boxes  # All bounding boxes detected
    
    vehicle_count = len(vehicle_boxes)  # Total number of detected vehicles
    return vehicle_count, vehicle_boxes
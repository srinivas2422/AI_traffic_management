from ultralytics import YOLO
import cv2  # OpenCV for image processing
import shutil
import os
from vehicle_detection import detect_vehicles

def detect_emergency_vehicles(
    frame_path,
    primary_model_path="../models/yolo11n.pt",
    fine_tuned_model_path="../models/yolo_fine_tuned/runs/detect/train/weights/best.pt",
    output_folder="./cropped_vehicles"
):
    """
    Detect emergency vehicles using a custom fine-tuned YOLO model for emergency classification first,
    and if detected, use a primary YOLO model for detailed vehicle detection and processing.
    
    Args:
        frame_path (str): Path to the image file.
        primary_model_path (str): Path to the primary YOLO model.
        fine_tuned_model_path (str): Path to the fine-tuned YOLO model for emergency vehicle detection.
        output_folder (str): Directory to save cropped images.

    Returns:
        str: "yes" if an emergency vehicle is detected, otherwise "no".
    """
    # Step 1: Load the fine-tuned YOLO model
    fine_tuned_model = YOLO(fine_tuned_model_path)
    
    # Step 2: Perform initial emergency vehicle detection on the entire frame
    results = fine_tuned_model(frame_path, conf=0.5)
    
    # Extract detected classes from the results
    detected_classes = []
    for result in results:
        if result.boxes and result.boxes.cls is not None:
            detected_classes.extend(
                [fine_tuned_model.names[int(cls)] for cls in result.boxes.cls]
            )
    
    # Step 3: Check if "emergency" is among the detected classes
    if "emergency" in detected_classes:
        # Step 4: Detect vehicles using the primary YOLO model
        _, vehicle_boxes = detect_vehicles(frame_path, primary_model_path)
        
        # Step 5: Read the image
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Prepare output directories
        os.makedirs(output_folder, exist_ok=True)
        emergency_folder = os.path.join(output_folder, "emergency_vehicles")
        non_emergency_folder = os.path.join(output_folder, "non_emergency_vehicles")
        os.makedirs(emergency_folder, exist_ok=True)
        os.makedirs(non_emergency_folder, exist_ok=True)
        
        # Process each vehicle bounding box
        emergency_vehicle_detected = False
        for idx, box in enumerate(vehicle_boxes):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert tensor to list and extract coordinates
            
            # Crop the vehicle image
            cropped_vehicle = frame[y1:y2, x1:x2]
            
            # Save the cropped vehicle image
            cropped_path = os.path.join(output_folder, f"vehicle_{idx+1}.jpg")
            cv2.imwrite(cropped_path, cv2.cvtColor(cropped_vehicle, cv2.COLOR_RGB2BGR))
            
            # Classify the cropped vehicle
            vehicle_results = fine_tuned_model(cropped_path, conf=0.5)
            
            # Extract detected classes for the cropped vehicle
            vehicle_detected_classes = []
            for vehicle_result in vehicle_results:
                if vehicle_result.boxes and vehicle_result.boxes.cls is not None:
                    vehicle_detected_classes.extend(
                        [fine_tuned_model.names[int(cls)] for cls in vehicle_result.boxes.cls]
                    )
            
            # Check if any detected class is "emergency"
            if "emergency" in vehicle_detected_classes:
                emergency_vehicle_detected = True
                final_save_path = os.path.join(emergency_folder, f"emergency_vehicle_{idx+1}.jpg")
            else:
                final_save_path = os.path.join(non_emergency_folder, f"non_emergency_vehicle_{idx+1}.jpg")
            
            # Move the cropped image to its final folder
            os.rename(cropped_path, final_save_path)
        
        shutil.rmtree(output_folder)
        return "yes" if emergency_vehicle_detected else "no"
    else:
        # No emergency vehicle detected in the frame
        return "no"

import cv2
import os
def extract_frames(video_path, output_folder, frame_rate):
    """Extract frames from a video and save them to the specified folder."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_interval = int(fps / frame_rate)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    print(f"Frames extracted to {output_folder}")

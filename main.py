import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # Load YOLO model
    model = YOLO('yolov8n.pt')  # Use nano model for efficiency
    
    # Corrected video path using raw string
    video_path = r'C:\jvn_codes\sen\object_thing\videoplayback.mp4'
    
    # Video capture
    cap = cv2.VideoCapture(video_path)
    
    # Tracking parameters
    tracked_objects = {}
    next_id = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform object detection
        results = model(frame)
        
        # Create a copy for drawing
        frame_draw = frame.copy()
        
        # Current frame's detections
        current_detections = []
        
        # Process detected objects
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                conf = float(box.conf)
                cls = int(box.cls)
                
                # Only process if confidence is above threshold
                if conf > 0.5:
                    # Store detection
                    current_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'cls': cls
                    })
        
        # Update tracked objects
        new_tracked_objects = {}
        
        # Match current detections with existing tracked objects
        for detection in current_detections:
            best_match = None
            best_dist = float('inf')
            
            # Find closest match to existing tracked objects
            for obj_id, prev_obj in tracked_objects.items():
                # Calculate center of previous and current bounding box
                prev_center = (
                    (prev_obj['bbox'][0] + prev_obj['bbox'][2]) / 2,
                    (prev_obj['bbox'][1] + prev_obj['bbox'][3]) / 2
                )
                curr_center = (
                    (detection['bbox'][0] + detection['bbox'][2]) / 2,
                    (detection['bbox'][1] + detection['bbox'][3]) / 2
                )
                
                # Calculate distance between centers
                dist = np.sqrt(
                    (prev_center[0] - curr_center[0])**2 + 
                    (prev_center[1] - curr_center[1])**2
                )
                
                # If close enough and same class, consider it a match
                if dist < 100 and prev_obj['cls'] == detection['cls']:
                    if dist < best_dist:
                        best_match = obj_id
                        best_dist = dist
            
            # Assign ID
            if best_match is not None:
                # Use existing ID
                new_tracked_objects[best_match] = {
                    'bbox': detection['bbox'],
                    'cls': detection['cls']
                }
            else:
                # New object, assign new ID
                new_tracked_objects[next_id] = {
                    'bbox': detection['bbox'],
                    'cls': detection['cls']
                }
                next_id += 1
        
        # Update tracked objects
        tracked_objects = new_tracked_objects
        
        # Draw bounding boxes and IDs
        for obj_id, obj in tracked_objects.items():
            x1, y1, x2, y2 = map(int, obj['bbox'])
            
            # Draw bounding box
            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ID and class
            label = f"ID:{obj_id}"
            cv2.putText(frame_draw, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Object Tracking', frame_draw)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
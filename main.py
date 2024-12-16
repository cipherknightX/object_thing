import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv
import json
import matplotlib.pyplot as plt

def get_class_name(model, class_id):
    """
    Retrieve class name from YOLO model

    Args:
        model (YOLO): Loaded YOLO model
        class_id (int): Numeric class ID

    Returns:
        str: Name of the detected class
    """
    return model.names[class_id]

def generate_general_report(json_file_path, output_dir):
    """
    Generate a general report summarizing all tracked objects in CSV format.

    Args:
        json_file_path (str): Path to the JSON file with tracking data.
        output_dir (str): Directory to save the general report.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    object_summary = {}
    
    for entry in data:
        class_name = entry['class']
        if class_name not in object_summary:
            object_summary[class_name] = 0
        object_summary[class_name] += 1

    # Save summary as a CSV file
    summary_file_path = os.path.join(output_dir, 'general_report.csv')
    with open(summary_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Object Class', 'Number of Appearances'])
        for class_name, count in object_summary.items():
            writer.writerow([class_name, count])

    print(f"General report saved to: {summary_file_path}")

def generate_specific_report(json_file_path, output_dir, target_class, fps):
    """
    Generate a specific report for a particular object class in CSV format, including total time and a graph.

    Args:
        json_file_path (str): Path to the JSON file with tracking data.
        output_dir (str): Directory to save the specific report.
        target_class (str): The class name to generate the report for.
        fps (int): Frames per second of the video.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    specific_data = []
    positions = []
    total_time = 0

    for entry in data:
        if entry['class'] == target_class:
            frame = entry['frame']
            time_seconds = frame / fps
            total_time += 1 / fps

            # Calculate position
            bbox = entry['bbox']
            center_x = (bbox['x1'] + bbox['x2']) / 2
            center_y = (bbox['y1'] + bbox['y2']) / 2

            # Determine position type
            position_type = "center" if 0.25 * 640 < center_x < 0.75 * 640 and 0.25 * 480 < center_y < 0.75 * 480 else "corner"
            positions.append(position_type)

            specific_data.append([frame, time_seconds, position_type])

    # Save specific report as a CSV file
    report_file_path = os.path.join(output_dir, f"{target_class}_report.csv")
    with open(report_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Frame', 'Time (s)', 'Position'])
        writer.writerows(specific_data)

    print(f"Specific report for {target_class} saved to: {report_file_path}")

    # Save total time in report
    total_time_report_path = os.path.join(output_dir, f"{target_class}_summary.csv")
    with open(total_time_report_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Object Class', 'Total Time (s)'])
        writer.writerow([target_class, round(total_time, 2)])

    print(f"Total time for {target_class} saved to: {total_time_report_path}")

    # Generate graph
    time_series = [entry[1] for entry in specific_data]
    plt.figure(figsize=(10, 6))
    plt.hist(time_series, bins=20, color='blue', alpha=0.7, label='Object Occurrence')
    plt.title(f"Time Distribution of {target_class}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.legend()

    graph_file_path = os.path.join(output_dir, f"{target_class}_graph.png")
    plt.savefig(graph_file_path)
    plt.close()

    print(f"Graph for {target_class} saved to: {graph_file_path}")

def main():
    # Load YOLO model
    model = YOLO('yolov8n.pt')  # Use nano model for efficiency

    # Corrected video path using raw string
    video_path = r'C:\jvn_codes\sen\object_thing\videoplayback.mp4'

    # Create output directory if it doesn't exist
    output_dir = r'C:\jvn_codes\sen\object_thing\output'
    os.makedirs(output_dir, exist_ok=True)

    # Video capture
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video and tracking data paths
    output_video_path = os.path.join(output_dir, 'tracked_video.mp4')
    output_csv_path = os.path.join(output_dir, 'object_tracking_data.csv')
    output_json_path = os.path.join(output_dir, 'object_tracking_data.json')

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Tracking parameters
    tracked_objects = {}
    tracking_data = []
    next_id = 0
    frame_count = 0

    # CSV file for logging
    csv_file = open(output_csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Object ID', 'Class', 'X1', 'Y1', 'X2', 'Y2', 'Confidence'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

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
                    'cls': detection['cls'],
                    'conf': detection['conf']  # Include confidence here
                }
            else:
                # New object, assign new ID
                new_tracked_objects[next_id] = {
                    'bbox': detection['bbox'],
                    'cls': detection['cls'],
                    'conf': detection['conf']  # Include confidence here
                }
                next_id += 1

        # Update tracked objects
        tracked_objects = new_tracked_objects

        # Draw bounding boxes and IDs, write to CSV
        for obj_id, obj in tracked_objects.items():
            x1, y1, x2, y2 = map(int, obj['bbox'])

            # Get class name
            class_name = get_class_name(model, obj['cls'])

            # Draw bounding box
            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw ID and class
            label = f"ID:{obj_id} {class_name}"
            cv2.putText(frame_draw, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write to CSV
            csv_writer.writerow([
                frame_count, obj_id, class_name, x1, y1, x2, y2, obj['conf']
            ])

            # Collect tracking data for JSON
            tracking_data.append({
                'frame': frame_count,
                'object_id': obj_id,
                'class': class_name,
                'bbox': {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                },
                'confidence': float(obj['conf'])
            })

        # Write frame to output video
        out.write(frame_draw)

        # Display the frame
        cv2.imshow('Object Tracking', frame_draw)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    out.release()
    csv_file.close()
    cv2.destroyAllWindows()

    # Write JSON output
    with open(output_json_path, 'w') as json_file:
        json.dump(tracking_data, json_file, indent=4)

    print(f"Tracked video saved to: {output_video_path}")
    print(f"Tracking data saved to: {output_csv_path}")
    print(f"Tracking data saved to JSON: {output_json_path}")

    # Generate reports
    generate_general_report(output_json_path, output_dir)
    generate_specific_report(output_json_path, output_dir, 'cell phone', fps)

if __name__ == "__main__":
    main()

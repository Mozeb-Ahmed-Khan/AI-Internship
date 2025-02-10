import cv2
import os
from ultralytics import YOLO

# Create directories for saving images and labels
os.makedirs("Images", exist_ok=True)
os.makedirs("Labels", exist_ok=True)

# Load a YOLOv8 model
model = YOLO(r'C:\Users\Mozeb.Khan\PycharmProjects\pythonProject\yolov8m.pt')  # Using yolov8m for better accuracy

def process_multiple_videos(video_paths):
    for video_index, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
        frame_count = 0
        image_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get frame dimensions
            image_height, image_width, _ = frame.shape

            # Process 1 frame per second
            if frame_count % fps == 0:
                # Save images with video index and image count
                image_path = f"Images/image_{video_index}_{image_count}.png"
                cv2.imwrite(image_path, frame)  # Save the frame as an image

                # Perform YOLOv8 inference with lower confidence threshold
                results = model.predict(frame, conf=0.25)

                # Save the YOLO output in label format with the same image count
                label_path = f"Labels/label_{video_index}_{image_count}.txt"
                with open(label_path, "w") as f:

                    for result in results:
                        for box in result.boxes:
                            # Extract box coordinates in (x_min, y_min, x_max, y_max) format
                            x_min, y_min, x_max, y_max = box.xyxy[0]

                            # Calculate mid_x, mid_y, pixel_width, pixel_height
                            mid_x = (x_min + x_max) / 2 / image_width  # Normalize by image width
                            mid_y = (y_min + y_max) / 2 / image_height  # Normalize by image height
                            pixel_width = (x_max - x_min) / image_width  # Normalize width
                            pixel_height = (y_max - y_min) / image_height  # Normalize height

                            # Write the class and bounding box in the new format
                            class_id = int(box.cls.item())  # Extract class id as an integer
                            f.write(f"{class_id} {mid_x:.6f} {mid_y:.6f} {pixel_width:.6f} {pixel_height:.6f}\n")


                image_count += 1  # Increment image_count after saving both image and label

            frame_count += 1

        cap.release()

# Function to automatically gather video files from a directory
def get_video_paths_from_directory(directory, video_extensions=['.mp4', '.avi', '.mkv', '.mov', '.flv']):
    video_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in video_extensions:
                video_paths.append(os.path.join(root, file))
    return video_paths

# Directory containing your video files
video_directory = r'C:\Users\Mozeb.Khan\PycharmProjects\pythonProject\Videos'  # Adjust the path to your video directory

# Automatically get all video paths from the directory
video_paths = get_video_paths_from_directory(video_directory)

# Process all videos
process_multiple_videos(video_paths)

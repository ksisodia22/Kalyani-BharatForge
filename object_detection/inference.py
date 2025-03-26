import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device:", self.device)
        self.model = self.load_model()

    def load_model(self):
        model = YOLO("best.pt")  # Path to your trained model
        model.fuse()  # Fuse model layers for optimization
        return model

    def predict(self, frame):
        results = self.model(frame)  # Perform inference
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []
        class_labels = ["carton","fire hydrant", "person", "shelf", "table"]
        # Loop over the results and extract bounding box information
        for result in results:
            boxes = result.boxes.cpu().numpy()  # Convert boxes to numpy array
            xyxys.append(boxes.xyxy)  # Get bounding boxes (x_min, y_min, x_max, y_max)
            confidences.append(boxes.conf)  # Get confidence scores for each box
            class_ids.append(boxes.cls)  # Get the class IDs for each box

        # Output bounding box information for each detected object
        for i, box in enumerate(xyxys[0]):
            x1, y1, x2, y2 = box  # Extract individual coordinates of the bounding box
            confidence = confidences[0][i]  # Confidence score for the box
            class_id = class_ids[0][i]  # Class ID of the detected object

            class_name = class_labels[int(class_id)] if int(class_id) < len(class_labels) else "Unknown"

            # Print bounding box coordinates, class ID, and confidence
            print(f"Bounding Box: ({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f})")
            print(f"Class ID: {class_id}, Class Name: {class_name}, Confidence: {confidence:.4f}")
            print("---")

        return frame, xyxys, confidences, class_ids


def main():
    obj_det = ObjectDetection(capture_index=0)

    # Read an input image (modify the path as needed)
    frame = cv2.imread("/home/khyati/image_372_jpg.rf.a30eef08108eb2e2d309d538956c0289.jpg")
    results = obj_det.predict(frame)
    
    # Process and extract bounding boxes and confidence
    processed_frame, bboxes, confidences, class_ids = obj_det.plot_bboxes(results, frame)

if __name__ == "__main__":
    main()

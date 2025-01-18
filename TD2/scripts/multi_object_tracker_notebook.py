import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

class MultiObjectTracker:
    def __init__(self, tracker_type="kcf"):
        self.trackers = cv2.legacy.MultiTracker_create()
        self.tracker_type = tracker_type

    def initialize_tracker(self, frame, rois):
        for roi in rois:
            tracker = self.get_tracker_by_type()
            self.trackers.add(tracker, frame, tuple(roi))

    def get_tracker_by_type(self):
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.legacy.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.legacy.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.legacy.TrackerTLD_create,
            "medianflow": cv2.legacy.TrackerMedianFlow_create,
            "mosse": cv2.legacy.TrackerMOSSE_create,
        }
        return OPENCV_OBJECT_TRACKERS[self.tracker_type]()

    def update(self, frame):
        success, boxes = self.trackers.update(frame)
        return success, boxes


def display_frame(frame):
    """Display the current frame using Matplotlib."""
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def select_rois(video_path):
    """Allow the user to select ROIs on the first frame of the video."""
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    rois = cv2.selectROIs("Select ROIs", frame, fromCenter=False, showCrosshair=True)
    cap.release()
    cv2.destroyAllWindows()
    return frame, rois


def save_results_to_csv(results, output_path):
    """Save tracking results to a CSV file."""
    df = pd.DataFrame(results, columns=["Frame", "ObjectID", "X", "Y", "W", "H"])
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def save_annotated_frame(frame, output_path, frame_number):
    """Save the current frame with annotations."""
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"frame_{frame_number:04d}.jpg")
    cv2.imwrite(output_file, frame)


def track_objects(video_path, tracker_type="kcf", output_dir="output"):
    """Track objects and save results."""
    cap = cv2.VideoCapture(video_path)
    mot = MultiObjectTracker(tracker_type)

    # Select ROIs
    print("[INFO] Selecting ROIs...")
    _, rois = select_rois(video_path)
    if len(rois) == 0:
        print("[ERROR] No ROIs selected. Exiting...")
        return

    print("[INFO] Initializing trackers...")
    success, first_frame = cap.read()
    mot.initialize_tracker(first_frame, rois)

    results = []
    frame_number = 0
    annotated_dir = os.path.join(output_dir, "frames")
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, boxes = mot.update(frame)

        if success:
            for idx, box in enumerate(boxes):
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Save results for each object
                results.append({"Frame": frame_number, "ObjectID": idx, "X": x, "Y": y, "W": w, "H": h})

        # Save annotated frame
        save_annotated_frame(frame, annotated_dir, frame_number)

        # Display frame (optional in Jupyter)
        display_frame(frame)

        frame_number += 1

    # Save results to CSV
    if results:
        csv_path = os.path.join(output_dir, "results.csv")
        save_results_to_csv(results, csv_path)
    else:
        print("[WARNING] No tracking data to save.")

    cap.release()

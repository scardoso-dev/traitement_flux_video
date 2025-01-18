import cv2
import matplotlib.pyplot as plt
import imutils
from IPython.display import HTML
import os

class ObjectTracker:
    def __init__(self, tracker_type="kcf"):
        self.tracker_type = tracker_type
        self.initBB = None
        self.fps = None
        self.tracker = self.initialize_tracker(tracker_type)

    def initialize_tracker(self, tracker_type):
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.legacy.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.legacy.TrackerBoosting_create,
            "mil": cv2.legacy.TrackerMIL_create,
            "tld": cv2.legacy.TrackerTLD_create,
            "medianflow": cv2.legacy.TrackerMedianFlow_create,
            "mosse": cv2.legacy.TrackerMOSSE_create
        }
        return OPENCV_OBJECT_TRACKERS[tracker_type]()

    def process_video(self, video_path=None):
        frames = []

        if video_path:
            vs = cv2.VideoCapture(video_path)
        else:
            vs = cv2.VideoCapture(0)

        while True:
            ret, frame = vs.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=500)

            if self.initBB is not None:
                (success, box) = self.tracker.update(frame)
                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        vs.release()
        return frames

    def play_video(self, frames, output_path="output_video.mp4"):
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        video_html = f"""
        <video width="600" controls>
            <source src="{output_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """
        return HTML(video_html)

    def start_tracking(self, video_path=None, output_path="output_video.mp4"):
        print("[INFO] Loading video...")
        frames = self.process_video(video_path)

        print("[INFO] Starting video playback...")
        return self.play_video(frames, output_path=output_path)

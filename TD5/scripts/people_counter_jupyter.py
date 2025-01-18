import cv2
import numpy as np
import imutils
import time
import os
import dlib
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import FPS
import matplotlib.pyplot as plt

def display_frame(frame):
    """Affiche une image dans le notebook avec Matplotlib."""
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()

def people_counter(prototxt, model, input_video=None, confidence=0.4, skip_frames=30):
    # Chargement du modèle MobileNet SSD
    print("[INFO] Loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Chargement de la vidéo ou activation de la webcam
    if input_video:
        print("[INFO] Opening video file...")
        vs = cv2.VideoCapture(input_video)
    else:
        print("[INFO] Starting video stream...")
        vs = cv2.VideoCapture(0)
        time.sleep(1.0)

    # Initialisation
    W, H = None, None
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}
    totalFrames = 0
    totalUp = 0
    totalDown = 0

    # Création du répertoire de sortie s'il n'existe pas
    output_dir = "output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fps = FPS().start()

    while True:
        # Lecture de la prochaine frame
        ret, frame = vs.read()
        if not ret:
            break

        # Redimensionnement de la frame
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            H, W = frame.shape[:2]

        status = "Waiting"
        rects = []

        # On effectue une détection toutes les N frames
        if totalFrames % skip_frames == 0:
            status = "Detecting"
            trackers = []
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence_score = detections[0, 0, i, 2]
                if confidence_score > confidence:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
        else:
            for tracker in trackers:
                status = "Tracking"
                tracker.update(rgb)
                pos = tracker.get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)
                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True

            trackableObjects[objectID] = to
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        info = [("Up", totalUp), ("Down", totalDown), ("Status", status)]
        for (i, (k, v)) in enumerate(info):
            text = f"{k}: {v}"
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Enregistrement de la frame annotée
        frame_output_path = os.path.join(output_dir, f"frame_{totalFrames:04d}.jpg")
        cv2.imwrite(frame_output_path, frame)

        # Affichage de la frame dans le notebook
        display_frame(frame)

        totalFrames += 1
        fps.update()

    fps.stop()
    print(f"[INFO] Elapsed time: {fps.elapsed():.2f}")
    print(f"[INFO] Approx. FPS: {fps.fps():.2f}")

    vs.release()
    cv2.destroyAllWindows()

    # On renvoie les statistiques finales
    return totalUp, totalDown

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

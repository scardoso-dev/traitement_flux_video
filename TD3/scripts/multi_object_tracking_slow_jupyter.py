import cv2
import dlib
import numpy as np
import imutils
from utils import FPS

# Définition des classes détectables par MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def multi_object_tracking_slow_jupyter(prototxt, model, video, confidence=0.2, output=None):
    # Chargement du modèle pré-entraîné
    print("[INFO] Chargement du modèle...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # Chargement de la vidéo
    print("[INFO] Chargement de la vidéo...")
    vs = cv2.VideoCapture(video)

    # Initialisation des trackers et du compteur FPS
    trackers = []
    labels = []
    fps = FPS().start()

    # Préparation de l'écriture de la vidéo de sortie
    writer = None
    output_size = None  # Pour stocker la taille des frames redimensionnées

    while True:
        grabbed, frame = vs.read()
        if frame is None:
            break

        # Redimentionnement de la frame
        frame = imutils.resize(frame, width=600)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Définition de la taille de sortie une seule fois après le redimensionnement
        if writer is None and output:
            output_size = (frame.shape[1], frame.shape[0])  # Largeur, Hauteur
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output, fourcc, 30, output_size, True)

        if len(trackers) == 0:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                if detections[0, 0, i, 2] > confidence:
                    idx = int(detections[0, 0, i, 1])
                    label = CLASSES[idx]
                    if label != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    t.start_track(rgb, rect)

                    trackers.append(t)
                    labels.append(label)

        else:
            for tracker, label in zip(trackers, labels):
                tracker.update(rgb)
                pos = tracker.get_position()

                (startX, startY, endX, endY) = (int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # On écrit la frame dans la vidéo de sortie
        if writer:
            writer.write(frame)

        fps.update()

    fps.stop()
    print("[INFO] Temps écoulé : {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS : {:.2f}".format(fps.fps()))

    if writer:
        writer.release()

    vs.release()

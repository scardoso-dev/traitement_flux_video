import cv2
import dlib
import numpy as np
import multiprocessing
from utils import FPS


def start_tracker(box, label, rgb, inputQueue, outputQueue):
    """
    Fonction pour initialiser et exécuter un tracker pour un objet donné.
    """
    tracker = dlib.correlation_tracker()
    rect = dlib.rectangle(*box)
    tracker.start_track(rgb, rect)

    while True:
        # On attend que le processus parent envoie un nouveau frame
        rgb = inputQueue.get()

        if rgb is None:
            break

        # Mise à jour du tracker et obtention de la position
        tracker.update(rgb)
        pos = tracker.get_position()
        outputQueue.put((label, (int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom()))))


def multi_object_tracking_fast_jupyter(prototxt, model, video, confidence=0.2, output=None):
    """
    Fonction pour le tracking rapide avec dlib et multiprocessing.
    """
    print("[INFO] Chargement du modèle...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    print("[INFO] Chargement de la vidéo...")
    vs = cv2.VideoCapture(video)

    # Initialisation des queues et les processus pour le tracking
    inputQueues = []
    outputQueues = []
    processes = []

    # Initialisation de FPS et writer
    fps = FPS().start()
    writer = None

    while True:
        grabbed, frame = vs.read()
        if frame is None:
            break

        frame = cv2.resize(frame, (600, int(frame.shape[0] * (600 / frame.shape[1]))))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if output is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output, fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

        if len(processes) == 0:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > confidence:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    box = box.astype("int")
                    label = CLASSES[idx]

                    # Crée des queues pour le processus
                    inputQueue = multiprocessing.Queue()
                    outputQueue = multiprocessing.Queue()
                    inputQueues.append(inputQueue)
                    outputQueues.append(outputQueue)

                    # Lance un processus pour le tracker
                    process = multiprocessing.Process(target=start_tracker,
                                                      args=(box, label, rgb, inputQueue, outputQueue))
                    process.daemon = True
                    process.start()
                    processes.append(process)

        else:
            # Envoi le frame à chaque processus via les queues
            for inputQueue in inputQueues:
                inputQueue.put(rgb)

            # Lecture des résultats des trackers
            for outputQueue in outputQueues:
                label, (startX, startY, endX, endY) = outputQueue.get()
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        if writer:
            writer.write(frame)

        fps.update()

    # Nettoyage
    fps.stop()
    print("[INFO] Temps écoulé : {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS : {:.2f}".format(fps.fps()))

    if writer:
        writer.release()

    vs.release()

    # On termine les processus
    for inputQueue in inputQueues:
        inputQueue.put(None)
    for process in processes:
        process.join()


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

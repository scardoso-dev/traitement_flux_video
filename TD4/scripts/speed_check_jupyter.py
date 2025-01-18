import cv2
import dlib
import math
from IPython.display import display, clear_output
from PIL import Image
import time

def show_frame(frame):
    """Affiche une image dans Jupyter Notebook."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    clear_output(wait=True)
    display(img)

def estimateSpeed(location1, location2, fps=30, ppm=8.8):
    """Estime la vitesse d'un objet en km/h."""
    x1, y1, w1, h1 = location1
    x2, y2, w2, h2 = location2
    distance_pixels = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    distance_meters = distance_pixels / ppm  # Conversion des pixels en mètres
    speed = distance_meters * fps * 3.6  # Conversion de m/s en km/h
    return speed

def trackMultipleObjects(video_path, cascade_path, width=1280, height=720):
    """
    Suit les véhicules dans une vidéo et estime leurs vitesses.

    :param video_path: Chemin de la vidéo
    :param cascade_path: Chemin du modèle Haar Cascade
    :param width: Largeur de la vidéo redimensionnée
    :param height: Hauteur de la vidéo redimensionnée
    """
    carCascade = cv2.CascadeClassifier(cascade_path)
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    
    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    while True:
        start_time = time.time()
        ret, image = video.read()
        if not ret:
            break
        
        image = cv2.resize(image, (width, height))
        resultImage = image.copy()
        
        frameCounter += 1
        
        carIDtoDelete = []

        # Mise à jour des trackers existants
        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)
            if trackingQuality < 7:
                carIDtoDelete.append(carID)
        
        # Suppression des trackers de mauvaise qualité
        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        # Détection des voitures tous les 10 frames
        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
            
            for (_x, _y, _w, _h) in cars:
                x, y, w, h = int(_x), int(_y), int(_w), int(_h)
                x_bar, y_bar = x + 0.5 * w, y + 0.5 * h
                
                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()
                    t_x, t_y = int(trackedPosition.left()), int(trackedPosition.top())
                    t_w, t_h = int(trackedPosition.width()), int(trackedPosition.height())
                    t_x_bar, t_y_bar = t_x + 0.5 * t_w, t_y + 0.5 * t_h
                    
                    if ((t_x <= x_bar <= (t_x + t_w)) and 
                        (t_y <= y_bar <= (t_y + t_h)) and 
                        (x <= t_x_bar <= (x + w)) and 
                        (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]
                    currentCarID += 1

        # Affichage des rectangles et calcul de vitesse
        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()
            t_x, t_y = int(trackedPosition.left()), int(trackedPosition.top())
            t_w, t_h = int(trackedPosition.width()), int(trackedPosition.height())
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            
            carLocation2[carID] = [t_x, t_y, t_w, t_h]
            
            if carLocation1.get(carID) and carLocation2.get(carID):
                [x1, y1, w1, h1] = carLocation1[carID]
                [x2, y2, w2, h2] = carLocation2[carID]
                carLocation1[carID] = [x2, y2, w2, h2]
                
                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if speed[carID] is None and y1 >= 275 and y1 <= 285:
                        speed[carID] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])
                    if speed[carID] is not None:
                        cv2.putText(resultImage, str(int(speed[carID])) + " km/hr", 
                                    (int(x1 + w1/2), int(y1-5)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Affichage de l'image dans le notebook
        show_frame(resultImage)

        if cv2.waitKey(33) == 27:
            break

    video.release()
    cv2.destroyAllWindows()

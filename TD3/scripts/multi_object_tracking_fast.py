from utils import FPS
import multiprocessing
import numpy as np
import argparse
import dlib
import cv2


def start_tracker(box, label, rgb, inputQueue, outputQueue):
	# missing algorithm
 
 
 
 
 
 

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


inputQueues = []
outputQueues = []

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
vs = cv2.VideoCapture(args["video"])
writer = None

fps = FPS().start()

if __name__ == '__main__':
	
	while True:
		(grabbed, frame) = vs.read()
	
		if frame is None:
			break
	
		(h, w) = frame.shape[:2]
		width=600
		r = width / float(w)
		dim = (width, int(h * r))
		frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(frame.shape[1], frame.shape[0]), True)
	
		
		if len(inputQueues) == 0:
			(h, w) = frame.shape[:2]
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
			net.setInput(blob)
			detections = net.forward()
			for i in np.arange(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]
				if confidence > args["confidence"]:
					idx = int(detections[0, 0, i, 1])
					label = CLASSES[idx]
					if CLASSES[idx] != "person":
						continue
					box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					(startX, startY, endX, endY) = box.astype("int")
					bb = (startX, startY, endX, endY)
	
				
					iq = multiprocessing.Queue()
					oq = multiprocessing.Queue()
					inputQueues.append(iq)
					outputQueues.append(oq)
					
				
					p = multiprocessing.Process(
						target=start_tracker,
						args=(bb, label, rgb, iq, oq))
					p.daemon = True
					p.start()
					
					cv2.rectangle(frame, (startX, startY), (endX, endY),
						(0, 255, 0), 2)
					cv2.putText(frame, label, (startX, startY - 15),
						cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	
		else:
			
			for iq in inputQueues:
				iq.put(rgb)
	
			for oq in outputQueues:
				
				(label, (startX, startY, endX, endY)) = oq.get()
	
				
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 255, 0), 2)
				cv2.putText(frame, label, (startX, startY - 15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	
		if writer is not None:
			writer.write(frame)
	
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	
		if key == 27:
			break

		fps.update()
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	
	if writer is not None:
		writer.release()

	cv2.destroyAllWindows()
	vs.release()
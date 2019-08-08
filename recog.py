from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import time
import cv2
import os
import logging

conf_rate = 75

num = pickle.load(open("data/number.pkl", "rb"))
names = pickle.load(open("data/names.pkl", "rb"))
passed = []
waiting = []
max = len(names)
info = []
dict = {}

def update(max, passed, waiting):
    info = []
    info.append(max)
    info.append(len(passed))
    info.append(max - len(passed))
    with open('data/info.txt', 'w') as f:
        for item in info:
            f.write("%s\n" % item)

def upper(name):
    first = name[0:1].upper()
    name = first + name[1:99]

logging.basicConfig(filename="history.log", level=logging.INFO)

protoPath = os.path.sep.join(["models", "deploy.prototxt"])
modelPath = os.path.sep.join(["models", "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

embedder = cv2.dnn.readNetFromTorch("models/openface_nn4.small2.v1.t7")

recognizer = pickle.loads(open("data/recognizer.pickle", "rb").read())
le = pickle.loads(open("data/le.pickle", "rb").read())

print("[INFO] starting live feed...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

fps = FPS().start()

while True:
	frame = vs.read()
	frame = np.flip(frame, axis=1)
	frame = imutils.resize(frame, height=600, width=800)
	(h, w) = frame.shape[:2]
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	detector.setInput(imageBlob)
	detections = detector.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			if fW < 20 or fH < 20:
				continue

			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			if proba * 100 > conf_rate:
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
				cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
				if name not in passed:
					passed.append(name)
				if name not in dict:
					dict[name] = time.strftime('%X %x')
					print("[INFO]", dict[name], name)
					log = dict[name] + " " + name
					logging.info(log)
			else:
				cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		for i in range(0, max):
			name = names[i]
			upper(name)
			if name in passed:
				if name in dict:
					if name in waiting:
						waiting.remove(name)
		for i in range(0, max):
			name = names[i]
			upper(name)
			if name not in passed:
				waiting.append(name)
	fps.update()
	waiting = isimler = list(dict.fromkeys(waiting))
	passed = isimler = list(dict.fromkeys(passed))
	update(max, passed, waiting)
	cv2.imshow("Live feed", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

fps.stop()
print("[INFO] total time: {:.2f} sec".format(fps.elapsed()))
print("[INFO] average FPS: {:.2f}".format(fps.fps()))
print("[INFO] wasn't in roll call: ")

for i in waiting:
	print(num[i] + " - " + i)


cv2.destroyAllWindows()
vs.stop()

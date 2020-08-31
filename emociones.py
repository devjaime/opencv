import cv2
import os
import numpy as np
import imutils

def emotionImage(emotion):
	# Emojis
	if emotion == 'Felicidad': image = cv2.imread('Emojis/felicidad.jpeg')
	if emotion == 'Enojo': image = cv2.imread('Emojis/enojo.jpeg')
	if emotion == 'Sorpresa': image = cv2.imread('Emojis/sorpresa.jpeg')
	if emotion == 'Tristeza': image = cv2.imread('Emojis/tristeza.jpeg')
	return image

# ----------- Métodos usados para el entrenamiento y lectura del modelo ----------
#method = 'EigenFaces'
#method = 'FisherFaces'
method = 'LBPH'

if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('modelo'+method+'.xml')
# --------------------------------------------------------------------------------

dataPath = '/Users/vn0i7o5/Desktop/opencv/data' #Cambia a la ruta donde hayas almacenado Data
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

cap = cv2.VideoCapture(cv2.CAP_AVFOUNDATION)
cap.set(3,640)
cap.set(4,480)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
	ret, frame = cap.read()
	if ret == False: break
	#frame =  imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = frame.copy()
	print(frame)
	nFrame = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)])
	#nFrame = frame
	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(200,200),interpolation= cv2.INTER_CUBIC)
		rostrogray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
		result = emotion_recognizer.predict(rostrogray)

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

		# EigenFaces
		if method == 'EigenFaces':
			if result[1] < 5700:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
		
		# FisherFace
		if method == 'FisherFaces':
			if result[1] < 500:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
		
		# LBPHFace
		if method == 'LBPH':
			if result[1] < 60:
				cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
				image = emotionImage(imagePaths[result[0]])
				nFrame = cv2.hconcat([frame,image])
			else:
				cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
				nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
	cv2.imshow('nFrame',nFrame)

	k = cv2.waitKey(1)
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
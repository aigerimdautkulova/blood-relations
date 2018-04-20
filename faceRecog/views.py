import cv2
import sys
import math
import dlib
import scipy
import imutils
import argparse
import numpy as np
import sklearn.datasets
import sklearn.metrics.pairwise
from skimage import io
from imutils import face_utils
from scipy.spatial import distance
from PIL import Image
from .settings import BASE_DIR
from django.shortcuts import render, redirect
from django.http import HttpResponse

from .models import DadInfo, MomInfo, ChildInfo
import time
import datetime
import os


# Ceate your views here.
def index(request):
    return render(request, 'index.html')
def errorImg(request):
    return render(request, 'error.html')


def detectImage(request):
	sp = dlib.shape_predictor(BASE_DIR+'/shape_predictor_68_face_landmarks.dat')
	facerec = dlib.face_recognition_model_v1(BASE_DIR+'/dlib_face_recognition_resnet_model_v1.dat')
	detector = dlib.get_frontal_face_detector()
	res = []

	if request.method == 'POST':
		ts = time.time()
		st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H_%M_%S')
		dirPath = BASE_DIR+'/static/uploadedimg/'+str(st)+'/'
		if not os.path.exists(dirPath):
			os.makedirs(dirPath)

		#ПАПА
		userImage1=request.FILES['userImage1']
		im1 = Image.open(userImage1)
		imgPath1 = str(dirPath)+str(userImage1)
		im1.save(imgPath1, 'JPEG')

		img = io.imread(imgPath1)
		dets = detector(img, 1)
		for k, d in enumerate(dets):
			left = d.left()
			right = d.right()
			top = d.top()
			bottom = d.bottom()
			face_image = img[top:bottom, left:right]

		face_image = imutils.resize(face_image, width=500)

		dets = detector(face_image, 1)
		for k, d in enumerate(dets):
			shape1 = sp(face_image, d)
		face_descriptor1 = facerec.compute_face_descriptor(face_image, shape1)
		#Dad's Mouth
		rects = detector(face_image, 1)
		for (i, rect) in enumerate(rects):
		    shape11 = sp(face_image, rect)
		    shape11 = face_utils.shape_to_np(shape11)
		(mStart1, mEnd1) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
		mouthPts1 = shape11[mStart1: mEnd1]
		#Dad's Nose
		rects1 = detector(face_image, 1)
		for (i, rect) in enumerate(rects1):
		    shape111 = sp(face_image, rect)
		    shape111 = face_utils.shape_to_np(shape111)
		(nStart1, nEnd1) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
		nosePts1 = shape111[nStart1: nEnd1]
		#Dad's Eye
		rects11 = detector(face_image, 1)
		for (i, rect) in enumerate(rects11):
		    shape1111 = sp(face_image, rect)
		    shape1111 = face_utils.shape_to_np(shape1111)
		(eStart1, eEnd1) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		eyePts1 = shape1111[eStart1: eEnd1]
		#Dad's Jaw
		rects111 = detector(face_image, 1)
		for (i, rect) in enumerate(rects111):
		    shape_11 = sp(face_image, rect)
		    shape_11 = face_utils.shape_to_np(shape_11)
		(jStart1, jEnd1) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
		jawPts1 = shape_11[jStart1: jEnd1]

		a = DadInfo(descriptor_dad=face_descriptor1, photo_dad=imgPath1)
		a.save()

        #Мама
		userImage2=request.FILES['userImage2']
		im2 = Image.open(userImage2)
		imgPath2 = str(dirPath)+str(userImage2)
		im2.save(imgPath2, 'JPEG')
		#Mom
		img2 = io.imread(imgPath2)
		dets = detector(img2, 1)
		for k, d in enumerate(dets):
		    left2 = d.left()
		    right2 = d.right()
		    top2 = d.top()
		    bottom2 = d.bottom() 
		    face_image2 = img2[top2:bottom2, left2:right2]
		face_image2 = imutils.resize(face_image2, width=500, height=800)

		dets = detector(face_image2, 1)
		for k, d in enumerate(dets):
		    shape_2 = sp(face_image2, d)

		face_descriptor2 = facerec.compute_face_descriptor(face_image2, shape_2)
		#Mom's Mouth
		rects2 = detector(face_image2, 1)
		for (i, rect) in enumerate(rects2):
		    shape2 = sp(face_image2, rect)
		    shape2 = face_utils.shape_to_np(shape2)
		(mStart2, mEnd2) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
		mouthPts2 = shape2[mStart2: mEnd2]
		#Mom's Nose
		rects_2 = detector(face_image2, 1)
		for (i, rect) in enumerate(rects_2):
		    shape22 = sp(face_image2, rect)
		    shape22 = face_utils.shape_to_np(shape22)
		(nStart2, nEnd2) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
		nosePts2 = shape22[nStart2: nEnd2]
		#Mom's Eye
		rects22 = detector(face_image2, 1)
		for (i, rect) in enumerate(rects22):
		    shape222 = sp(face_image2, rect)
		    shape222 = face_utils.shape_to_np(shape222)
		(eStart2, eEnd2) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		eyePts2 = shape222[eStart2: eEnd2]
		#Mom's Jaw
		rects_22 = detector(face_image2, 1)
		for (i, rect) in enumerate(rects_22):
		    shape_22 = sp(face_image2, rect)
		    shape_22 = face_utils.shape_to_np(shape_22)
		(jStart2, jEnd2) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
		jawPts2 = shape_22[jStart2: jEnd2]


		b = MomInfo(descriptor_mom=face_descriptor2, photo_mom=imgPath2)
		b.save()

        #Ребенок
		userImage3 = request.FILES['userImage3']
		im3 = Image.open(userImage3)
		imgPath3 = str(dirPath)+str(userImage3)
		im3.save(imgPath3, 'JPEG')

		#Child
		img3 = io.imread(imgPath3)

		dets = detector(img3, 1)
		for k, d in enumerate(dets):
		    left3 = d.left()
		    right3 = d.right()
		    top3 = d.top()
		    bottom3 = d.bottom() 
		    face_image3 = img3[top3:bottom3, left3:right3]
		face_image3 = imutils.resize(face_image3, width=500, height=800)

		dets = detector(face_image3, 1)
		for k, d in enumerate(dets):
		    shape_3 = sp(face_image3, d)

		face_descriptor3 = facerec.compute_face_descriptor(face_image3, shape_3)
		#Child's Mouth
		rects3 = detector(face_image3, 1)
		for (i, rect) in enumerate(rects3):
		    shape3 = sp(face_image3, rect)
		    shape3 = face_utils.shape_to_np(shape3)
		(mStart3, mEnd3) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
		mouthPts3 = shape3[mStart3: mEnd3]
		#child's Nose
		rects_3 = detector(face_image3, 1)
		for (i, rect) in enumerate(rects_3):
		    shape33 = sp(face_image3, rect)
		    shape33 = face_utils.shape_to_np(shape33)
		(nStart3, nEnd3) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
		nosePts3 = shape33[nStart3: nEnd3]
		#Child's Eye
		rects33 = detector(face_image3, 1)
		for (i, rect) in enumerate(rects33):
		    shape333 = sp(face_image3, rect)
		    shape333 = face_utils.shape_to_np(shape333)
		(eStart3, eEnd3) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		eyePts3 = shape333[eStart3: eEnd3]
		#Child's Jaw
		rects_33 = detector(face_image3, 1)
		for (i, rect) in enumerate(rects_33):
		    shape_33 = sp(face_image3, rect)
		    shape_33 = face_utils.shape_to_np(shape_33)
		(jStart3, jEnd3) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
		jawPts3 = shape_33[jStart3: jEnd3]


		c = ChildInfo(descriptor_child=face_descriptor3, photo_child=imgPath3, dad=a, mom=b)
		c.save()

		#Общий результат
		a = distance.euclidean(face_descriptor1, face_descriptor3)
		b = distance.euclidean(face_descriptor2, face_descriptor3)
		if (a<b): res.append("Общими чертами ребенок больше похож на отца")
		else: res.append("Общими чертами ребенок больше похож на мать")


		#Рот
		results1 = []
		for p1, p2 in zip(mouthPts1, mouthPts3):
		    results1.append(math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) ))
		avg1 = np.average(results1)

		results2 = []
		for p3, p4 in zip(mouthPts2, mouthPts3):
		    results2.append(math.sqrt( ((p3[0]-p4[0])**2)+((p3[1]-p4[1])**2) ))
		avg2 = np.average(results2)

		if (avg1<avg2): res.append("Губы от отца")
		else: res.append("Губы от матери")

		for p5 in mouthPts3:
		    bottomLeftCornerOfText = (p5[0]-125,p5[1])

		font        = cv2.FONT_HERSHEY_SIMPLEX
		fontScale   = 1
		fontColor   = (255,255,255)
		lineType    = 2
		    
		if (avg1<avg2): cv2.putText(face_image3,'Dad',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
		else: cv2.putText(face_image3,'Mom',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)


		#Нос
		results11 = []
		for r1, r2 in zip(nosePts1, nosePts3):
		    results11.append(math.sqrt( ((r1[0]-r2[0])**2)+((r1[1]-r2[1])**2) ))
		avg11 = np.average(results11)

		results22 = []
		for r3, r4 in zip(nosePts2, nosePts3):
		    results22.append(math.sqrt( ((r3[0]-r4[0])**2)+((r3[1]-r4[1])**2) ))
		avg22 = np.average(results22)

		if (avg11<avg22): res.append("Нос от отца")
		else: res.append("Нос от матери")

		for j5 in nosePts3:
		    bottomLeftCornerOfText = (j5[0]-150,j5[1])

		font        = cv2.FONT_HERSHEY_SIMPLEX
		fontScale   = 1
		fontColor   = (255,255,255)
		lineType    = 2
		    
		if (avg11<avg22): cv2.putText(face_image3,'Dad',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
		else: cv2.putText(face_image3,'Mom',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)


		#Глаза
		results111 = []
		for s1, s2 in zip(eyePts1, eyePts3):
		    results111.append(math.sqrt( ((s1[0]-s2[0])**2)+((s1[1]-s2[1])**2) ))
		avg111 = np.average(results111)

		results222 = []
		for s3, s4 in zip(eyePts2, eyePts3):
		    results222.append(math.sqrt( ((s3[0]-s4[0])**2)+((s3[1]-s4[1])**2) ))
		avg222 = np.average(results222)

		if (avg111<avg222): res.append("Глаза от отца")
		else: res.append("Глаза от матери")

		for t5 in eyePts3:
		    bottomLeftCornerOfText = (t5[0],t5[1]-100)

		font        = cv2.FONT_HERSHEY_SIMPLEX
		fontScale   = 1
		fontColor   = (255,255,255)
		lineType    = 2
		    
		if (avg111<avg222): cv2.putText(face_image3,'Dad',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
		else: cv2.putText(face_image3,'Mom',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)


		#Подбородок
		results_1 = []
		for k1, k2 in zip(jawPts1, jawPts3):
		    results_1.append(math.sqrt( ((k1[0]-k2[0])**2)+((k1[1]-k2[1])**2) ))
		avg_1 = np.average(results_1)

		results_2 = []
		for l3, l4 in zip(jawPts2, jawPts3):
		    results_2.append(math.sqrt( ((l3[0]-l4[0])**2)+((l3[1]-l4[1])**2) ))
		avg_2 = np.average(results_2)

		if (avg_1<avg_2): res.append("Подбородок от отца")
		else: res.append("Подбородок от матери")

		for g5 in jawPts3:
		    bottomLeftCornerOfText = (g5[0]-100,g5[1]+150)

		font        = cv2.FONT_HERSHEY_SIMPLEX
		fontScale   = 1
		fontColor   = (255,255,255)
		lineType    = 2

		if (avg_1<avg_2): cv2.putText(face_image3,'Dad',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
		else: cv2.putText(face_image3,'Mom',bottomLeftCornerOfText,font,fontScale,fontColor,lineType)

		imgpathresult = BASE_DIR+'/static/result/'+str(userImage3)
		name1 = str(userImage3)
		res.append(name1)
		cv2.imwrite(imgpathresult, cv2.cvtColor(face_image3, cv2.COLOR_RGB2BGR))

	return render(request, 'index.html', {'result1':res})
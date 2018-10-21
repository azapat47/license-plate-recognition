import numpy as np
import cv2
import time
import math
import glob, os

import keras_model_digits
import keras_model_digits

from keras_model_digits import ConvNet_digits
from keras_model_alpha import ConvNet_alpha

from numpy.random import seed
from tensorflow import set_random_seed

contador = 0
placas = set([])

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect(c):
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    else:
        shape = "circle"
    return shape, len(approx)

def yellow(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 0, 130])
    upper_yellow = np.array([35,255,255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask

def es_placa(placa):
    cont = 0
    global placas
    for i in placa:
        if i.isdigit() or i.isalpha():
            cont += 1
    if cont >= 5 :
        print(cont)
        placa = [x for x in placa if x != "."]
        plac = ''.join(placa)
        placas.add(plac)
        print(plac)
    else:
        print("no placa")


def segmentacion(img):
    arr = np.array(img)    
    fin = int(len(arr[0])/6)
    inicio = 0
    incremento = fin
    margen = int(len(arr)/8)
    sizey = len(arr)
    global contador
    #contador += 1
    placa = []
    for i in range(6):
        new = arr[0+margen:sizey-2*margen,inicio:fin]
        inicio = fin
        fin += incremento
        if(i>0 and i<3):
            print("l")
            #placa.append(#)
            #model_letras(new)
        else:
            print("l")
            #placa.append(#)
            #model_numero(new)
        #cv2.imwrite("./placas/img"+str(contador)+str(i)+".jpg", new)
    #es_placa(placa)
    #cv2.waitKey(0)
    
def retornaplaca(img_bin, img_resized, contours):
    mini = 1000
    mayor = 0
    maxi = 1000000
    en_cuenta = []
    for h,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area>mini and area < maxi:
            x,y,w,h = cv2.boundingRect(cnt)
            tmp = img_bin[y:y+h, x:x+w]
            tmp2 = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(tmp2)

            pts = np.array([(box[1][0], box[1][1]), (box[2][0], box[2][1]), (box[3][0], box[3][1]), (box[0][0], box[0][1])])
            tmp = four_point_transform(img_bin, pts)

            _, conts, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cont in conts:
                area = cv2.contourArea(cont)
                if area>mini and area < maxi:
                    form, corners = detect(cont)
                    x,y,w,h = cv2.boundingRect(cont)
                    mask = tmp[y:y+h, x:x+w]
                    if ((form == 'rectangle' and w > h*1.5) or (corners > 4 and corners < 7 and w > h*1.5)):
                        box = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(box)
                        pts = np.array([(box[1][0], box[1][1]), (box[2][0], box[2][1]), (box[3][0], box[3][1]), (box[0][0], box[0][1])])
                        box = four_point_transform(img_resized, pts)
                        box_bin = four_point_transform(img_bin, pts)
                        suma = np.sum([box_bin > 0])
                        total = box_bin.shape[0]*box_bin.shape[1]
                        if(suma/total > 0.6):
                            box_b = yellow(box)
                            _, conto, _ = cv2.findContours(box_b,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                            #cv2.imshow('placa', box)
                            #cv2.imshow('placab', box_b)
                            segmentacion(box_b)
                            #cv2.waitKey()


def preprocesamiento(img):
    img_resized = img
    #cv2.imwrite('imagen.jpg', img_resized)
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    img_red = img_lab[:, :, 2]
    _,img_bin1 = cv2.threshold(img_red.copy(),140,255,cv2.THRESH_BINARY)
    mini = np.array([15, 45, 45])
    maxi = np.array([30, 255, 255])
    img_bin = cv2.inRange(img_hsv, mini, maxi)
    img_bin = cv2.bitwise_and(img_bin, img_bin1)
    #cv2.imshow('original', cv2.resize(img_resized, (1000, 500)))
    #cv2.waitKey()
    im2, contours, hierarchy = cv2.findContours(img_bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    retornaplaca(img_bin, img_resized, contours)

def w_file():
    path = './deadpool.csv'
    plad = open(path,'w')
    global placas
    for i in placas:
        plad.write(str(i)+"\n")
        print(str(i)+"\n")
    plad.close()
    

    
def video():
    videoFile = "../../video.mp4"
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5)/10 #frame rate
    #print(frameRate)
    # Read until video is completed
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        #print(frameId)
        ret, frame = cap.read()
        if ret == True:
            if (frameId % frameRate) == 0:
                #frame = cv2.resize(frame, (1000, 500))
                preprocesamiento(frame)
                #cv2.imshow('Frame',frame)
                
                # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
    cap.release()
    w_file()

#video()

def main():
	seed(0)
	set_random_seed(0)
	# 0-0 98% 80 iters 1.0 drop_p -> bad z, bad g

	alpha_convNet = ConvNet_alpha()
	alpha_convNet.build_graph()
	if (os.path.exists('alpha_model.h5')):
		alpha_convNet.restore_model()
	else:
		alpha_convNet.train()
		alpha_convNet.save_model()
		
	seed(0)
	set_random_seed(0)
	# 0-0 98% 80 iters 1.0 drop_p -> bad z, bad g

	digit_convNet = ConvNet_digits()
	digit_convNet.build_graph()
	if (os.path.exists('alpha_model.h5')):
		alpha_convNet.restore_model()
	else:
		digits_convNet.train()
		digits_convNet.save_model()

	preds_folder = "data/preds/"
	prediction = alpha_convNet.predict_from_file(preds_folder,"z.jpeg")
	print(prediction)
	preds_folder = "data/preds/"
	prediction = digit_convNet.predict_from_file(preds_folder,"8.jpeg")
	print(prediction)

main()

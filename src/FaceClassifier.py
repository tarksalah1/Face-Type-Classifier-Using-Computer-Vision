from dis import dis
from PyQt5 import QtWidgets as qtw
from PyQt5 import uic
import cv2
from numpy.fft import fft2, fftshift
import numpy as np
from prometheus_client import Counter
from viewer import Viewer
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from skimage.color import rgb2gray
from skimage.io import imread
import mediapipe as mp
import math

class FaceClassifier(qtw.QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi("src/ui/face_type_classification.ui", self)
        self.face = Viewer()
        self.face_layout.addWidget(self.face)
        self.landMarks = Viewer()
        self.landmark_layout.addWidget(self.landMarks)
        self.apply_btn.clicked.connect(self.showLandmarks)

        #initailazation 
        self.faceImage = None
        self.faceImageCopy = None
        self.landmarksInitialized = False
        self.dic = {}

        self.mpFaceMesh = mp.solutions.face_mesh
        self.connectionLeftEye= self.mpFaceMesh.FACEMESH_LEFT_EYE
        self.connectionOval = self.mpFaceMesh.FACEMESH_FACE_OVAL
        self.connectionRightEye = self.mpFaceMesh.FACEMESH_RIGHT_EYE
        self.connectionRightEyeBrow = self.mpFaceMesh.FACEMESH_RIGHT_EYEBROW
        self.connectionleftEyeBrow = self.mpFaceMesh.FACEMESH_LEFT_EYEBROW
        self.connectionLips = self.mpFaceMesh.FACEMESH_LIPS

        self.leftEyeIndices = self.getUnique(self.connectionLeftEye)
        self.ovalIndices = self.getUnique(self.connectionOval)
        self.rightEyeIndices = self.getUnique(self.connectionRightEye)
        self.rightEyeBrowIndices = self.getUnique(self.connectionRightEyeBrow)
        self.leftEyeBrowIndices = self.getUnique(self.connectionleftEyeBrow)
        self.lipsIndices = self.getUnique(self.connectionLips)

        self.centerPointIndex=5
        self.buttonPointIndex =152
        self.topPointIndex = 10
        self.reCorrectedIndex = {338,10,109,67,103,54,297,332,284,251,21}

        self.faceLength =[152,10]
        self.foreheadLength = [251,21]
        self.jawlineLength = [152,132]
        self.chinWidth = [379,150]
        self.checkBoneWidth = [300,70]
        self.angleA1 = [234,172]
        self.angleA2 = [150,172]
        self.angleA3 = [150,152]

    def loadFace(self, image_path):
        self.faceImage = cv2.imread(image_path)
        self.faceImage = cv2.cvtColor(self.faceImage, cv2.COLOR_BGR2RGB)
        self.clear(self.face)
        self.clear(self.landMarks)
        self.draw(self.face,self.faceImage)
        self.faceImageCopy =self.faceImage.copy()
        self.landmarksInitialized = False
        self.initiateLandmarks(self.faceImageCopy)
        self.calculateFeature(self.faceLength, vertical = True,name="faceLength")
        self.calculateFeature(self.foreheadLength ,horizonal=True,name = "foreheadLength")
        self.calculateFeature(self.chinWidth ,horizonal=True,name = "chinWidth")
        self.calculateFeature(self.checkBoneWidth, horizonal=True,oval = False,name = "checkBoneWidth")
        self.calculateFeature(self.jawlineLength, tilted=True,name = "jawlineLength")
        self.calculateFeature(self.angleA1, angle=True,name = "angleA1")
        self.calculateFeature(self.angleA2, angle=True,name = "angleA2")
        self.calculateFeature(self.angleA3, angle=True,name = "angleA3")




        print(self.dic)
        print("angle in radian")


        
        

    def draw(self,layout,image):
        layout.draw_image(image)

    def clear(self,layout):
        layout.clear_canvans()

    def initiateLandmarks(self,faceImage):
        faceMesh = self.mpFaceMesh.FaceMesh()
        results = faceMesh.process(faceImage)
        for faceLms in results.multi_face_landmarks:
            landMarks = faceLms.landmark

            #for oval points
            self.ovalDic=self.localizingPointsInImage(faceImage,landMarks,self.ovalIndices,list(self.connectionOval))
            self.leftEyeDic= self.localizingPointsInImage(faceImage,landMarks,self.leftEyeIndices,list(self.connectionLeftEye))
            self.rightEyeDic =self.localizingPointsInImage(faceImage,landMarks,self.rightEyeIndices,list(self.connectionRightEye))
            self.rightEyeBrowDic =self.localizingPointsInImage(faceImage,landMarks,self.rightEyeBrowIndices,list(self.connectionRightEyeBrow))
            self.leftEyeBrowDic =self.localizingPointsInImage(faceImage,landMarks,self.leftEyeBrowIndices,list(self.connectionleftEyeBrow))
            self.lipsDic = self.localizingPointsInImage(faceImage,landMarks,self.lipsIndices,list(self.connectionLips))
            self.centerDic =  self.localizingPointsInImage(faceImage,landMarks,{self.centerPointIndex},[(5,5)])
            self.buttonDic = { self.buttonPointIndex:self.ovalDic[ self.buttonPointIndex]}
            self.topDic={self.topPointIndex  :self.ovalDic[self.topPointIndex]}
            self.tneButtonhalfLengthOfFace = self.buttonDic[self.buttonPointIndex][1]-self.centerDic[self.centerPointIndex][1]
            self.tneTophalfLengthOfFace = self.centerDic[self.centerPointIndex][1]-self.topDic[self.topPointIndex][1]
            if(self.tneButtonhalfLengthOfFace>self.tneTophalfLengthOfFace):
                ynew= self.centerDic[self.centerPointIndex][1]-self.tneButtonhalfLengthOfFace
                correction = self.topDic[self.topPointIndex][1]-ynew
                for index in self.reCorrectedIndex:
                    self.ovalDic[index]= (self.ovalDic[index][0],self.ovalDic[index][1]-correction)

            for index in self.ovalDic:
                cv2.circle(faceImage,self.ovalDic[index],2,(255,255,255),-1)
            for con in self.connectionOval:
                cv2.line(faceImage,self.ovalDic[con[0]],self.ovalDic[con[1]],(250,0,240),1)
            
            self.landmarksInitialized = True

    def showLandmarks(self):
        if self.landmarksInitialized:
            self.draw(self.landMarks,self.faceImageCopy)  
            # fig, axs = plt.subplots(figsize=(15, 10))
            # axs.imshow(self.faceImage,cmap=plt.get_cmap('gray'))
            # plt.show() 

    def localizingPointsInImage(self,img,lms,indices,connections):
        dic = {}
        for index in indices:     
            ih, iw, ic = img.shape
            x, y = int(lms[index].x * iw), int(lms[index].y * ih)
            dic[index]=(x,y)
            cv2.circle(img,(x,y),2,(255,255,255),-1)
            # cv2.putText(self.faceImage, str(index), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

        for con in connections:  
            cv2.line(img,dic[con[0]],dic[con[1]],(250,250,240),1)  
        return dic
        
    def getUnique(self,c):
        tempList = list(c)
        tempSet = set()
        for i in tempList:
            tempSet.add(i[0])
            tempSet.add(i[1])
        return list(tempSet)   

    def calculateFeature(self,listIndices, vertical= False, horizonal = False ,angle = False ,tilted= False ,oval = True,name=""):
        if self.initiateLandmarks:
            if not oval :       
                distance = np.abs(self.leftEyeBrowDic[listIndices[0]][0]- self.rightEyeBrowDic[listIndices[1]][0])
                self.dic[name] = distance   

            else:
                if vertical:
                    distance = np.abs(self.ovalDic[listIndices[0]][1]- self.ovalDic[listIndices[1]][1])
                    self.dic[name] = distance

                if horizonal:
                    distance = np.abs(self.ovalDic[listIndices[0]][0]- self.ovalDic[listIndices[1]][0])
                    self.dic[name] = distance   

                if tilted:
                    distance1 = np.abs(self.ovalDic[listIndices[0]][0]- self.ovalDic[listIndices[1]][0])
                    distance2 = np.abs(self.ovalDic[listIndices[0]][1]- self.ovalDic[listIndices[1]][1])
                    distance = np.sqrt(distance1*distance1+distance2 *distance2)
                    self.dic[name] = distance

                if angle:
                    tanAngle = (self.ovalDic[listIndices[0]][1]- self.ovalDic[listIndices[1]][1])/ (self.ovalDic[listIndices[0]][0]- self.ovalDic[listIndices[1]][0]) 
                    self.dic[name] = math.atan(tanAngle)
                    

            


                     





    # def binarizing_image(self):
    #     _,self.Gray_img_binary=cv2.threshold(self.imag1,self.threshold_value.value(),255,cv2.THRESH_BINARY)
    #     self.thershold_value.setText(str(self.threshold_value.value()))  
    #     self.draw(self.Binary_image ,self.Gray_img_binary)
    #     # self.Gray_img=self.Gray_img_binary

    

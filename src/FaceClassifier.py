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

class FaceClassifier(qtw.QWidget):
    def __init__(self):
        super().__init__()

        uic.loadUi("src/ui/face_type_classification.ui", self)
        self.face = Viewer()
        self.face_layout.addWidget(self.face)
        self.landMarks = Viewer()
        self.landmark_layout.addWidget(self.landMarks)
        # self.segmentation = Viewer()
        # self.segmentation_layout.addWidget(self.segmentation)
        # self.heatmap = Viewer()
        # self.heatmap_layout.addWidget(self.heatmap)
        
        # self.threshold_value.valueChanged.connect(self.binarizing_image)

    def loadFace(self, image_path):
        self.Gray_img = cv2.imread(image_path,0)
        self.imag1=self.Gray_img
        self.draw(self.face,self.Gray_img)
        
        

    def draw(self,layout,image):
        layout.draw_image(image)

    def clear(self,layout):
        layout.clear_canvans()

    # def binarizing_image(self):
    #     _,self.Gray_img_binary=cv2.threshold(self.imag1,self.threshold_value.value(),255,cv2.THRESH_BINARY)
    #     self.thershold_value.setText(str(self.threshold_value.value()))  
    #     self.draw(self.Binary_image ,self.Gray_img_binary)
    #     # self.Gray_img=self.Gray_img_binary

    


            


          

                  
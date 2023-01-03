import sys
from PyQt5.QtWidgets import QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2
import math
import csv
from matplotlib.figure import Figure
# import mediapipe as mp
from PyQt5 import QtWidgets
from time import time
import dlib
import numpy as np
from imutils import face_utils
import matplotlib.pyplot as plt

from DetermineShape import DetermineShape
from dlibalgorism import Dlibalgorism
from mainWindow import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        ####################################################################
        # Variables:
        self.images_data = {}
        self.images_counter = 0
        self.FaceLength = 0
        self.Forehead = 0
        self.cheekbone_width = 0
        self.jawline_length = 0
        self.ChinWidth = 0
        self.angle_A1 = 0
        self.angle_A2 = 0
        self.angle_A3 = 0

        # tab1:
        self.original_image_figure_1 = Figure()
        self.original_image_canvas_1 = FigureCanvas(
            self.original_image_figure_1)
        self.ui.verticalLayout.addWidget(self.original_image_canvas_1)

        self.face_detected_image_figure = Figure()
        self.face_detected_image_canvas = FigureCanvas(
            self.face_detected_image_figure)
        self.ui.verticalLayout_2.addWidget(self.face_detected_image_canvas)

        # tab2
        self.original_image_figure_2 = Figure()
        self.original_image_canvas_2 = FigureCanvas(
            self.original_image_figure_2)
        self.ui.original_image_layout.addWidget(self.original_image_canvas_2)

        self.processed_image_figure = Figure()
        self.processed_image_canvas = FigureCanvas(self.processed_image_figure)
        self.ui.processed_image_layout.addWidget(self.processed_image_canvas)

        self.face_shape_figure = Figure()
        self.face_shape_canvas = FigureCanvas(self.face_shape_figure)
        self.ui.verticalLayout_5.addWidget(self.face_shape_canvas)

        self.ui.browse_button_1.clicked.connect(self.browse_image)

        # self.ui.browse_button_2.clicked.connect(self.browse_image)
        self.ui.apply_button.clicked.connect(self.dlib_library_algortihm)
        self.ui.face_type_bt.clicked.connect(self.face_type_classification)

        self.ui.detect_button.clicked.connect(self.hogDetectFaces)
        self.ui.switch_tabs_button.clicked.connect(self.switch_tab)

        # Initialization
        self.image_file_path = None
        self.original_image = None
        self.original_axes_1 = None
        self.original_axes_2 = None
        self.processed_axes = None
        self.image_with_lines = None
        self.face_detected_axes = None
        self.image_with_lines_axes = None
        self.predictor81 = None
        self.processed_image = None
        self.processed_image_feature = None
        self.face_detected_image = None
        self.featureDic = None
        self.ratioFeatureDic = None
        self.final_face_type = "Not detected"
        self.dlibAlgorism = None
        self.determineShape = None
        self.done = "false"
        self.detected = "false"

    def raise_error(self, error_message):
        QtWidgets.QMessageBox.about(
            self, "Error", error_message)

    def browse_image(self):
        try:
            self.done = "false"
            self.detected = "false"
            self.image_file_path = QFileDialog.getOpenFileName(
                filter="Image (*.png *.jpg *.jpeg)")[0]

            self.original_image = plt.imread(self.image_file_path)
            self.plot_original_image_in_tab1()
            self.plot_original_image_in_tab2()
        except Exception as e:
            print(e)

    def plot_original_image_in_tab1(self):
        try:
            self.original_axes_1 = self.original_image_figure_1.gca()
            self.original_axes_1.get_xaxis().set_visible(False)
            self.original_axes_1.get_yaxis().set_visible(False)
            self.original_axes_1.imshow(self.original_image)
            self.original_image_canvas_1.draw()
            self.original_image_canvas_1.flush_events()
        except Exception as e:
            print(e)

    def plot_original_image_in_tab2(self):
        try:
            self.original_axes_2 = self.original_image_figure_2.gca()
            self.original_axes_2.get_xaxis().set_visible(False)
            self.original_axes_2.get_yaxis().set_visible(False)
            self.original_axes_2.imshow(self.original_image)
            self.original_image_canvas_2.draw()
            self.original_image_canvas_2.flush_events()
        except Exception as e:
            print(e)

    def plot_processed_image(self):
        try:
            self.processed_axes = self.processed_image_figure.gca()
            self.processed_axes.get_xaxis().set_visible(False)
            self.processed_axes.get_yaxis().set_visible(False)
            self.processed_axes.imshow(self.processed_image)
            self.processed_image_canvas.draw()
            self.processed_image_canvas.flush_events()
        except Exception as e:
            print(e)

    def plot_face_shape_image(self):
        try:
            self.image_with_lines_axes = self.face_shape_figure.gca()
            self.image_with_lines_axes.get_xaxis().set_visible(False)
            self.image_with_lines_axes.get_yaxis().set_visible(False)
            self.image_with_lines_axes.imshow(self.image_with_lines)
            self.face_shape_canvas.draw()
            self.face_detected_image_canvas.flush_events()
        except Exception as e:
            print(e)

    def plot_faces_detected_image(self):
        try:
            self.face_detected_axes = self.face_detected_image_figure.gca()
            self.face_detected_axes.get_xaxis().set_visible(False)
            self.face_detected_axes.get_yaxis().set_visible(False)
            self.face_detected_axes.imshow(self.face_detected_image)
            self.face_detected_image_canvas.draw()
            self.face_detected_image_canvas.flush_events()
        except Exception as e:
            print(e)

    def switch_tab(self):
        try:
            if self.detected == "false":
                self.hogDetectFaces()
                self.original_axes_2 = self.original_image_figure_2.gca()
                self.original_axes_2.get_xaxis().set_visible(False)
                self.original_axes_2.get_yaxis().set_visible(False)
                self.original_axes_2.imshow(self.face_detected_image)
                self.original_image_canvas_2.draw()
                self.original_image_canvas_2.flush_events()
            else:
                self.ui.tabWidget.setCurrentIndex(1)
                self.original_axes_2 = self.original_image_figure_2.gca()
                self.original_axes_2.get_xaxis().set_visible(False)
                self.original_axes_2.get_yaxis().set_visible(False)
                self.original_axes_2.imshow(self.face_detected_image)
                self.original_image_canvas_2.draw()
                self.original_image_canvas_2.flush_events()
            if self.detected == "true":
                self.ui.tabWidget.setCurrentIndex(1)

        except Exception as e:
            print(e)

    def dlib_library_algortihm(self):
        try:
            if self.detected == "false":
                self.hogDetectFaces()
            if self.detected == "true":
                self.dlibAlgorism = Dlibalgorism(self.original_image)
                self.determineShape = DetermineShape()
                self.featureDic, self.ratioFeatureDic, self.processed_image, self.image_with_lines = self.dlibAlgorism.dlibAlgortihm()
                self.final_face_type = self.determineShape.deterShape(self.ratioFeatureDic)
                self.plot_processed_image()
                self.plot_face_shape_image()
                self.done = "true"
                self.set_face_shape_type()


        except Exception as e:
            print(e)

    def face_type_classification(self):
        if self.done == "false":
            self.dlib_library_algortihm()
        else:
            self.ui.tabWidget.setCurrentIndex(2)

    def set_face_shape_type(self):
        try:
            self.ui.label_6.setText(self.final_face_type)
        except Exception as e:
            print(e)

    # for Face Detection

    def hogDetectFaces(self):
        try:
            hog_face_detector = dlib.get_frontal_face_detector()

            height, width, _ = self.original_image.shape

            self.face_detected_image = self.original_image.copy()

            imgRGB = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

            start = time()

            results = hog_face_detector(imgRGB, 0)

            end = time()

            for bbox in results:
                x1 = bbox.left()
                y1 = bbox.top()
                x2 = bbox.right()
                y2 = bbox.bottom()

                cv2.rectangle(self.face_detected_image, pt1=(x1, y1), pt2=(x2, y2),
                              color=(0, 255, 0), thickness=width // 200)

            cv2.putText(self.face_detected_image, text='Time taken: ' + str(round(end - start, 2)) + ' Seconds.',
                        org=(10, 65),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=width // 700, color=(255, 250, 255),
                        thickness=width // 500)

            self.plot_faces_detected_image()
            self.detected = "true"

        except Exception as e:
            self.detected = "false"
            self.raise_error("Face can not be detected change the type of the image or check for face "
                             "existence")
            print(e)

        # return self.face_detected_image, results

    ###################################################################################################################################
    ###################################################################################################################################
    # for Mediapipe Algorithm

    def GetUnique(self, c):
        tempList = list(c)
        tempSet = set()
        for i in tempList:
            tempSet.add(i[0])
            tempSet.add(i[1])
        return list(tempSet)

    def LocalizingPointsInImage(self, img, lms, indices, connections):
        dic = {}
        for index in indices:
            ih, iw, ic = img.shape
            x, y = int(lms[index].x * iw), int(lms[index].y * ih)
            dic[index] = (x, y)
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1)

        for con in connections:
            cv2.line(img, dic[con[0]], dic[con[1]], (250, 250, 240), 1)
        return dic

    ###################################################################################################################################


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Curve Fitting and Interpolation")
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

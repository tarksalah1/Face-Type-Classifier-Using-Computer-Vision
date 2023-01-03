import dlib
from imutils import face_utils
import cv2
import math
import numpy as np


class Dlibalgorism:
    def __init__(self, image):
        self.featureDic = {}
        self.ratioFeatureDic = {"A1": 0, "A2": 0, "A3": 0, "R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0, "R6": 0,
                                "R7": 0, "R8": 0, "R9": 0, "R10": 0}

        self.predictor81 = None
        self.processed_image = None
        self.processed_image_feature = None
        self.pointFeature = []
        self.points = None
        self.original_image = image

    def dlibAlgortihm(self):
        try:
            self.predictor81 = dlib.shape_predictor(
                './Data/shape_predictor_81_face_landmarks.dat')

            landmarks = self.facial_landmarks(self.original_image)
            if landmarks is not None:
                # Find eye landmarks used for alignment
                eyePoints = (landmarks[39], landmarks[42])

                # Align face
                self.original_image = self.align_face(
                    self.original_image, eyePoints)
                # newLandMarks
                landmarks = self.facial_landmarks(self.original_image)

                # Draw landmarks
                self.processed_image = self.drawPoints(
                    self.original_image, landmarks)

                # calculateParameters
                self.calculate_parameters(landmarks)
                # print(self.featureDic)
                # print(self.ratioFeatureDic)

            return self.featureDic, self.ratioFeatureDic, self.processed_image, self.drawFeature(
                self.original_image.copy())

        except Exception as e:
            print(e)

    def drawFeature(self, processed_image):
        for index in self.pointFeature:
            cv2.line(processed_image, self.points[index[0]], self.points[index[1]], (0, 250, 0), 2)
            cv2.circle(processed_image, self.points[index[0]], 2, (255, 0, 0), -1)
            cv2.circle(processed_image, self.points[index[1]], 2, (255, 0, 0), -1)
        cv2.line(processed_image, self.points[14], (self.points[14][0] - 50, self.points[14][1]), (0, 0, 255), 2)
        cv2.line(processed_image, self.points[8], (self.points[8][0], self.points[8][1] - 50), (0, 0, 255), 2)
        cv2.line(processed_image, self.points[8], (self.points[10]), (0, 0, 255), 2)
        cv2.line(processed_image, self.points[14], (self.points[12]), (0, 0, 255), 2)

        return processed_image

    def facial_landmarks(self, image):
        # Use dlib 68 & 81 to predict landmarks points coordinates
        detector = dlib.get_frontal_face_detector()

        # Grayscale image
        try:
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except:
            grayscale_image = image

        # array of rectangles surrounding faces detected
        rectangles = detector(grayscale_image, 1)

        # If at least one face is detected, find its landmarks
        if len(rectangles) > 0:

            faceLandmarks = self.predictor81(grayscale_image, rectangles[0])

            faceLandmarks = face_utils.shape_to_np(faceLandmarks)
            return faceLandmarks

        # No faces found
        else:
            return None

    def align_face(self, image, eyePoints):
        # Get left eye & right eye coordinates
        leftEyeX, leftEyeY = eyePoints[0]
        rightEyeX, rightEyeY = eyePoints[1]

        # Calculate angle of rotation & origin point
        angle = math.atan((leftEyeY - rightEyeY) /
                          (leftEyeX - rightEyeX)) * (180 / math.pi)

        origin_point = tuple(np.array(image.shape[1::-1]) / 2)

        # Rotate using rotation matrix
        rot_mat = cv2.getRotationMatrix2D(origin_point, angle, 1.0)
        result = cv2.warpAffine(
            image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def drawPoints(self, image, points, pointColor=(255, 255, 255), lineColor=(255, 255, 255), pointThickness=7,
                   lineThickness=1):
        counter = 0
        for i in points:
            if counter == 6 or counter == 10:
                pointColor = (255, 0, 0)
            else:
                pointColor = (255, 255, 255)
            counter += 1
            x, y = i
            image = cv2.circle(image, (x, y), radius=0,
                               color=pointColor, thickness=pointThickness)
        return image

    def calculate_parameters(self, points):
        self.points = points
        D1 = round(math.sqrt(
            (points[2][0] - points[14][0]) ** 2 + (points[2][1] - points[14][1]) ** 2))
        self.pointFeature.append([2, 14])
        self.featureDic["D1"] = D1

        D2 = round(math.sqrt(
            (points[75][0] - points[79][0]) ** 2 + (points[75][1] - points[79][1]) ** 2))
        self.pointFeature.append([76, 79])
        self.featureDic["D2"] = D2

        center_forehead = (round((points[69][0] + points[72][0]) / 2),
                           round((points[69][1] + points[72][1]) / 2))

        D3 = round(math.sqrt(
            (center_forehead[0] - points[8][0]) ** 2 + (center_forehead[1] - points[8][1]) ** 2))
        self.pointFeature.append([8, 71])
        self.featureDic["D3"] = D3

        D4 = round(math.sqrt(
            (points[8][0] - points[12][0]) ** 2 + (points[8][1] - points[12][1]) ** 2))
        self.pointFeature.append([12, 8])
        self.featureDic["D4"] = D4

        D5 = round(math.sqrt(
            (points[4][0] - points[12][0]) ** 2 + (points[4][1] - points[12][1]) ** 2))
        self.pointFeature.append([4, 12])
        self.featureDic["D5"] = D5

        D6 = round(math.sqrt(
            (points[6][0] - points[10][0]) ** 2 + (points[6][1] - points[10][1]) ** 2))
        self.pointFeature.append([6, 10])
        self.featureDic["D6"] = D6

        D7 = round(math.sqrt(
            (points[7][0] - points[9][0]) ** 2 + (points[7][1] - points[9][1]) ** 2))
        self.pointFeature.append([7, 9])
        self.featureDic["D7"] = D7

        angleA1 = -1 * math.atan(
            (points[10][1] - points[8][1]) / (points[10][0] - points[8][0])) * (180 / math.pi)
        self.featureDic["angleA1"] = 90 - int(angleA1)
        self.ratioFeatureDic["A1"] = 90 - int(angleA1)
        angleA2 = -1 * math.atan(
            (points[12][1] - points[8][1]) / (points[12][0] - points[8][0])) * (180 / math.pi)
        # # print("angle A2: ", self.angle_A2)
        self.featureDic["angleA2"] = 90 - int(angleA2)
        self.ratioFeatureDic["A2"] = 90 - int(angleA2)
        angleA3 = math.atan(
            (points[2][1] - points[3][1]) / (points[2][0] - points[3][0])) * (180 / math.pi)
        self.featureDic["angleA3"] = int(angleA3)
        self.ratioFeatureDic["A3"] = int(angleA3)

        R1 = D2 / D1 * 100
        self.ratioFeatureDic["R1"] = R1

        R2 = D1 / D3 * 100
        self.ratioFeatureDic["R2"] = R2

        R3 = D2 / D3 * 100
        self.ratioFeatureDic["R3"] = R3

        R4 = D1 / D5 * 100
        self.ratioFeatureDic["R4"] = R4

        R5 = D6 / D5 * 100
        self.ratioFeatureDic["R5"] = R5

        R6 = D4 / D6 * 100
        self.ratioFeatureDic["R6"] = R6

        R7 = D6 / D1 * 100
        self.ratioFeatureDic["R7"] = R7

        R8 = D5 / D2 * 100
        self.ratioFeatureDic["R8"] = R8

        R9 = D4 / D5 * 100
        self.ratioFeatureDic["R9"] = R9

        R10 = D7 / D6 * 100
        self.ratioFeatureDic["R10"] = R10

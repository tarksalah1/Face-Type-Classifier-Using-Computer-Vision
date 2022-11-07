import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
import numpy as np

def GetUnique(c):
    tempList = list(c)
    tempSet = set()
    for i in tempList:
        tempSet.add(i[0])
        tempSet.add(i[1])
    return list(tempSet)

def LocalizingPointsInImage(img,lms,indices,connections):
    dic = {}
    for index in indices:     
        ih, iw, ic = img.shape
        x, y = int(lms[index].x * iw), int(lms[index].y * ih)
        dic[index]=(x,y)
        cv2.circle(img,(x,y),2,(255,255,255),-1)

    for con in connections:  
        cv2.line(img,dic[con[0]],dic[con[1]],(250,250,240),1)  
    return dic
    
img = cv2.imread("face.jpg") 
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgCopy=img.copy()

mpFaceMesh = mp.solutions.face_mesh
connectionLeftEye= mpFaceMesh.FACEMESH_LEFT_EYE
connectionOval = mpFaceMesh.FACEMESH_FACE_OVAL
connectionRightEye = mpFaceMesh.FACEMESH_RIGHT_EYE
connectionRightEyeBrow = mpFaceMesh.FACEMESH_RIGHT_EYEBROW
connectionleftEyeBrow = mpFaceMesh.FACEMESH_LEFT_EYEBROW
connectionLips = mpFaceMesh.FACEMESH_LIPS
# connectionBaseLine = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,8),(8,9),(9,9)]

leftEyeIndices = GetUnique(connectionLeftEye)
ovalIndices = GetUnique(connectionOval)
rightEyeIndices = GetUnique(connectionRightEye)
rightEyeBrowIndices = GetUnique(connectionRightEyeBrow)
leftEyeBrowIndices = GetUnique(connectionleftEyeBrow)
lipsIndices = GetUnique(connectionLips)
# baseLineIndices = {0,1,2,3,4,5,6,8,9}
centerpoint=5
buttonPoint =152
topPoint = 10
reCorrectedIndex = {338,10,109,67,103,54,297,332,284,251,21}
faceMesh = mpFaceMesh.FaceMesh()
results = faceMesh.process(imgCopy)
for faceLms in results.multi_face_landmarks:
    landMarks = faceLms.landmark

    #for oval points
    ovalDic=LocalizingPointsInImage(imgCopy,landMarks,ovalIndices,list(connectionOval))
    leftEyeDic= LocalizingPointsInImage(imgCopy,landMarks,leftEyeIndices,list(connectionLeftEye))
    rightEyeDic =LocalizingPointsInImage(imgCopy,landMarks,rightEyeIndices,list(connectionRightEye))
    rightEyeBrowDic =LocalizingPointsInImage(imgCopy,landMarks,rightEyeBrowIndices,list(connectionRightEyeBrow))
    leftEyeBrowDic =LocalizingPointsInImage(imgCopy,landMarks,leftEyeBrowIndices,list(connectionleftEyeBrow))
    lipsDic = LocalizingPointsInImage(imgCopy,landMarks,lipsIndices,list(connectionLips))
    centerDic =  LocalizingPointsInImage(imgCopy,landMarks,{centerpoint},[(5,5)])
    buttonDic = {buttonPoint:ovalDic[buttonPoint]}
    topDic={topPoint :ovalDic[topPoint]}
    tneButtonhalfLengthOfFace = buttonDic[buttonPoint][1]-centerDic[centerpoint][1]
    tneTophalfLengthOfFace = centerDic[centerpoint][1]-topDic[topPoint][1]
    # ynew= centerDic[centerpoint][1]-tneButtonhalfLengthOfFace
    # cv2.line(imgCopy,centerDic[centerpoint],(topDic[topPoint][0],ynew),(250,0,240),1)  

    print(tneButtonhalfLengthOfFace)
    print(tneTophalfLengthOfFace)
    # cv2.line(imgCopy,buttonDic[buttonPoint],centerDic[centerpoint],(250,0,240),1)  
    # cv2.line(imgCopy,centerDic[centerpoint],topDic[topPoint],(250,0,240),1)  
    if(tneButtonhalfLengthOfFace>tneTophalfLengthOfFace):
        ynew= centerDic[centerpoint][1]-tneButtonhalfLengthOfFace
        correction = topDic[topPoint][1]-ynew
        for index in reCorrectedIndex:
            ovalDic[index]= (ovalDic[index][0],ovalDic[index][1]-correction)

    for index in ovalDic:
        cv2.circle(img,ovalDic[index],2,(255,255,255),-1)
    for con in connectionOval:
        cv2.line(imgCopy,ovalDic[con[0]],ovalDic[con[1]],(250,0,240),1)
    fig, axs = plt.subplots(figsize=(15, 10))
    axs.imshow(imgCopy,cmap=plt.get_cmap('gray'))
    plt.show()




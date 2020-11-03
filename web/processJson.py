import base64
import numpy as np
import cv2
import sys
from algoModule.estimateAgeGender.algo import estimateAgeGender
from algoModule.embedFace.myMath import distance

def base642np(data):
    imgName = list(data.keys())[0]
    base64Str = list(data.values())[0]
    base64Str = base64Str[22:] if base64Str.startswith('data:image') else base64Str
    img_data = base64.b64decode(base64Str)
    nparr = np.frombuffer(img_data, np.uint8)
    nparr3D = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return nparr3D

def estimateAgeGender(data, ageGenderEstimater):
    nparr3D = base642np(data)
    age, gender = ageGenderEstimater.estimateAgeGenderbyArray(nparr3D)
    return age, gender

def embedFace(data, faceEmbedder):
    nparr3D = base642np(data)
    imgName = list(data.keys())[0]
    vec = faceEmbedder.embedFace(nparr3D)
    return imgName, vec.ravel()
    
def compareSimilarity(data):
    feature1 = list(data.values())[0]
    feature2 = list(data.values())[1]
    arrFeature1 = np.array(feature1)
    arrFeature2 = np.array(feature2)
    dist = distance.distance(arrFeature1, arrFeature2)
    return dist

import cv2
from CamToPix import GetFaceImg
from CamToPix import MLAPI

def main():
    HaarFace=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')#to detect face
    CamObj=GetFaceImg.Camera(HaarFace,MLAPI)
    if CamObj.restart==1:
        main()
main()

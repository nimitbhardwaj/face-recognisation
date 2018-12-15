import cv2

def showImage(gray_img):
    cv2.imshow('Test Imag', gray_img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
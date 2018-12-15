import cv2
from utilities import utils
import time

class Detector(object):
    def __init__(self):
        self.classifier = cv2.CascadeClassifier("assets/lbp_cascade_train.xml")
    def start(self, img=None):
        if img == None:
            for frame in self.__getCam():
                if type(frame) == int:
                    break
                else:
                    frame = self.getClassifiedImg(frame)
                    cv2.imshow('Image', frame)

        else:
            img = cv2.imread(img)
            img = self.getClassifiedImg(img)
            utils.showImage(img)

    def getClassifiedImg(self, img, onlyFace=False):
        grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self.classifier.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=5)
        if onlyFace:
            if len(faces) == 0:
                return None, None, False
            x, y, w, h = faces[0]
            return img[y:y+w, x:x+h], faces[0], True
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return img

    def __getCam(self):
        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            yield frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        yield 0